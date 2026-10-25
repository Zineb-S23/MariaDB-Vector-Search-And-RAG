"""
Unit tests for the WikiRAG fake embedding and repository implementations.

These tests verify deterministic embeddings, article persistence,
cosine similarity, hybrid filtering, link resolution, and validation.

All tests run in memory without MariaDB or a downloaded AI model.
"""

import math
from dataclasses import replace
from datetime import datetime

import pytest
from fakes import FakeEmbedder, InMemoryRepository

from wikirag.config import EMBEDDING_DIM
from wikirag.errors import (
    EmbeddingDimensionError,
    SchemaMismatchError,
    ValidationError,
)
from wikirag.models import ArticleRecord, ChunkRecord, SearchFilters


def _vector(axis: int) -> tuple[float, ...]:
    """
    Create a unit vector with one nonzero component.

    Args:
        axis (int): Index of the component containing 1.0.

    Returns:
        tuple[float, ...]: A 384-dimensional unit vector.
    """

    return tuple(1.0 if index == axis else 0.0 for index in range(EMBEDDING_DIM))


def _article(
    article_id: int,
    title: str,
    *,
    topic: str,
    categories: tuple[str, ...],
    length_chars: int,
    revised_at: datetime,
    linked_titles: tuple[str, ...] = (),
) -> ArticleRecord:
    """
    Build valid article metadata for repository tests.

    Args:
        article_id (int): Unique Wikipedia-style article identifier.
        title (str): Article title.
        topic (str): Corpus topic.
        categories (tuple[str, ...]): Article categories.
        length_chars (int): Cleaned article length.
        revised_at (datetime): Revision timestamp.
        linked_titles (tuple[str, ...]): Titles linked from this article.

    Returns:
        ArticleRecord: A valid immutable article record.
    """

    return ArticleRecord(
        article_id=article_id,
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title}",
        revision_id=article_id * 10,
        fingerprint=f"{article_id:064x}",
        revised_at=revised_at,
        length_chars=length_chars,
        fetched_at=datetime(2026, 10, 1),
        topic=topic,
        categories=categories,
        linked_titles=linked_titles,
    )


def _seed_repository(repo: InMemoryRepository) -> None:
    """
    Populate the repository with three predictable articles.

    The articles have different topics, categories, lengths,
    timestamps, and embeddings so every filter can be tested.

    Args:
        repo (InMemoryRepository): Empty repository to populate.

    Returns:
        None: Inserts the articles and resolves their links.
    """

    articles = (
        _article(
            1,
            "Alpha",
            topic="physics",
            categories=("Astronomy",),
            length_chars=400,
            revised_at=datetime(2026, 5, 1),
            linked_titles=("Beta", "Outside"),
        ),
        _article(
            2,
            "Beta",
            topic="physics",
            categories=("Computing",),
            length_chars=200,
            revised_at=datetime(2025, 5, 1),
        ),
        _article(
            3,
            "Gamma",
            topic="biology",
            categories=("Science",),
            length_chars=600,
            revised_at=datetime(2026, 6, 1),
        ),
    )

    for axis, article in enumerate(articles):
        chunk = ChunkRecord(
            chunk_index=0,
            text=f"Example content for {article.title}",
            embedding=_vector(axis),
        )

        assert repo.ingest_article(article, [chunk]) == "inserted"

    # The link to "Outside" is ignored because it is not in the corpus.
    assert repo.resolve_links() == 1


def test_fake_embedder_is_deterministic(
    fake_embedder: FakeEmbedder,
) -> None:
    """
    Verify embedding dimensions, reproducibility, and normalization.

    Args:
        fake_embedder (FakeEmbedder): Shared fake embedding fixture.

    Returns:
        None: Passes when repeated inputs produce identical unit vectors.
    """

    first = fake_embedder.embed_query("What is MariaDB?")
    second = fake_embedder.embed_query("What is MariaDB?")

    assert fake_embedder.dim == EMBEDDING_DIM
    assert len(first) == EMBEDDING_DIM
    assert first == second
    assert all(math.isfinite(value) for value in first)
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)


@pytest.mark.parametrize("invalid_text", ["", "   ", 123])
def test_fake_embedder_rejects_invalid_queries(
    fake_embedder: FakeEmbedder,
    invalid_text: object,
) -> None:
    """
    Verify that invalid query text is rejected.

    Args:
        fake_embedder (FakeEmbedder): Fake embedding fixture.
        invalid_text (object): Invalid query supplied by pytest.

    Returns:
        None: Passes when ValidationError is raised.
    """

    with pytest.raises(ValidationError):
        fake_embedder.embed_query(invalid_text)


def test_fake_embedder_handles_batches(
    fake_embedder: FakeEmbedder,
) -> None:
    """
    Verify that batch embeddings preserve input order.

    Args:
        fake_embedder (FakeEmbedder): Fake embedding fixture.

    Returns:
        None: Passes when batch embeddings match individual embeddings.
    """

    texts = ["First article", "Second article"]

    vectors = fake_embedder.embed_texts(texts, batch_size=2)

    assert fake_embedder.embed_texts([]) == []
    assert len(vectors) == 2
    assert vectors[0] == fake_embedder.embed_query(texts[0])
    assert vectors[1] == fake_embedder.embed_query(texts[1])


def test_repository_insert_skip_and_update(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify insertion, fingerprint-based skipping, and updates.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.

    Returns:
        None: Passes when ingestion outcomes and counts are correct.
    """

    article = _article(
        1,
        "Alpha",
        topic="physics",
        categories=("Astronomy",),
        length_chars=100,
        revised_at=datetime(2026, 5, 1),
    )

    first_chunk = ChunkRecord(0, "Original text", _vector(0))

    assert fake_repo.ingest_article(article, [first_chunk]) == "inserted"
    assert fake_repo.ingest_article(article, [first_chunk]) == "skipped"

    # A changed fingerprint tells the repository to replace old chunks.
    updated_article = replace(
        article,
        fingerprint="b" * 64,
        title="Alpha Updated",
    )
    updated_chunk = ChunkRecord(0, "Updated text", _vector(1))

    assert (
        fake_repo.ingest_article(
            updated_article,
            [updated_chunk],
        )
        == "updated"
    )

    assert fake_repo.count_articles() == 1
    assert fake_repo.count_chunks() == 1
    assert fake_repo.get_article(1).title == "Alpha Updated"
    assert fake_repo.get_fingerprint(1) == "b" * 64


def test_repository_search_ranks_cosine_distance(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify cosine-distance ordering and similarity scores.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.

    Returns:
        None: Passes when exact matches rank first with correct scores.
    """

    _seed_repository(fake_repo)

    hits = fake_repo.search_chunks(_vector(0), top_k=3)

    assert [hit.article_id for hit in hits] == [1, 2, 3]
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].distance == pytest.approx(1.0)

    # The score contract is always score = 1 - cosine distance.
    for hit in hits:
        assert hit.score == pytest.approx(1.0 - hit.distance)


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (SearchFilters(topics=("BIOLOGY",)), [3]),
        (SearchFilters(categories=("Astronomy",)), [1]),
        (SearchFilters(article_ids=(2,)), [2]),
        (SearchFilters(exclude_article_ids=(1, 2)), [3]),
        (SearchFilters(min_length_chars=400), [3]),
        (SearchFilters(edited_since=datetime(2026, 1, 1)), [1, 3]),
        (SearchFilters(linked_from_article_id=1), [2]),
        (SearchFilters(max_distance=0.0), [1]),
        (SearchFilters(categories=("Astronomy", "Science")), [1, 3]),
    ],
)
def test_repository_hybrid_filters(
    fake_repo: InMemoryRepository,
    filters: SearchFilters,
    expected_ids: list[int],
) -> None:
    """
    Verify every supported hybrid-search filter.

    Args:
        fake_repo (InMemoryRepository): Repository fixture.
        filters (SearchFilters): Filters supplied by pytest.
        expected_ids (list[int]): Expected matching article identifiers.

    Returns:
        None: Passes when filtered search returns the expected articles.
    """

    _seed_repository(fake_repo)

    hits = fake_repo.search_chunks(
        _vector(0),
        top_k=10,
        filters=filters,
    )

    assert [hit.article_id for hit in hits] == expected_ids


def test_repository_combines_filters_with_and(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify that different filter fields are combined using AND.

    Args:
        fake_repo (InMemoryRepository): Repository fixture.

    Returns:
        None: Passes when conflicting filters produce no results.
    """

    _seed_repository(fake_repo)

    filters = SearchFilters(
        topics=("physics",),
        min_length_chars=500,
    )

    assert (
        fake_repo.search_chunks(
            _vector(0),
            filters=filters,
        )
        == []
    )


def test_repository_respects_top_k(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify that search returns no more than the requested result count.

    Args:
        fake_repo (InMemoryRepository): Repository fixture.

    Returns:
        None: Passes when the result limit is respected.
    """

    _seed_repository(fake_repo)

    assert len(fake_repo.search_chunks(_vector(0), top_k=2)) == 2


def test_repository_resolves_and_deletes_links(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify link resolution and removal of deleted article references.

    Args:
        fake_repo (InMemoryRepository): Repository fixture.

    Returns:
        None: Passes when deleting an article also removes its links.
    """

    _seed_repository(fake_repo)

    assert [article.article_id for article in fake_repo.get_linked_articles(1)] == [2]

    assert fake_repo.delete_article(2) is True
    assert fake_repo.delete_article(2) is False
    assert fake_repo.get_linked_articles(1) == []
    assert fake_repo.count_articles() == 2
    assert fake_repo.count_chunks() == 2


def test_repository_returns_neighbor_chunks(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify that neighboring chunks remain within the same article.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.

    Returns:
        None: Passes when chunks are returned in their original order.
    """

    article = _article(
        1,
        "Alpha",
        topic="physics",
        categories=(),
        length_chars=300,
        revised_at=datetime(2026, 5, 1),
    )

    chunks = [ChunkRecord(index, f"Text {index}", _vector(index)) for index in range(3)]

    fake_repo.ingest_article(article, chunks)

    # The centre chunk has one neighbor on either side.
    neighbors = fake_repo.get_neighbor_chunks(2, window=1)

    assert [chunk.chunk_index for chunk in neighbors] == [0, 1, 2]
    assert [chunk.chunk_index for chunk in fake_repo.get_neighbor_chunks(2, window=0)] == [1]


@pytest.mark.parametrize(
    "invalid_vector",
    [
        (0.0,) * EMBEDDING_DIM,
        (1.0,) * (EMBEDDING_DIM - 1),
        (float("nan"),) + (0.0,) * (EMBEDDING_DIM - 1),
        (float("inf"),) + (0.0,) * (EMBEDDING_DIM - 1),
    ],
)
def test_repository_rejects_invalid_vectors(
    fake_repo: InMemoryRepository,
    invalid_vector: tuple[float, ...],
) -> None:
    """
    Verify rejection of malformed or unusable query embeddings.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.
        invalid_vector (tuple[float, ...]): Invalid vector from pytest.

    Returns:
        None: Passes when EmbeddingDimensionError is raised.
    """

    with pytest.raises(EmbeddingDimensionError):
        fake_repo.search_chunks(invalid_vector)


def test_repository_rejects_incompatible_embedding_metadata(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify that embeddings from incompatible models cannot be mixed.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.

    Returns:
        None: Passes when incompatible model metadata is rejected.
    """

    fake_repo.ensure_embedding_meta("model-A", EMBEDDING_DIM)
    fake_repo.ensure_embedding_meta("model-A", EMBEDDING_DIM)

    with pytest.raises(SchemaMismatchError):
        fake_repo.ensure_embedding_meta("model-B", EMBEDDING_DIM)

    with pytest.raises(SchemaMismatchError):
        fake_repo.ensure_embedding_meta("model-A", 256)


@pytest.mark.parametrize("invalid_top_k", [0, 101, True])
def test_repository_rejects_invalid_top_k(
    fake_repo: InMemoryRepository,
    invalid_top_k: object,
) -> None:
    """
    Verify that invalid result limits are rejected.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.
        invalid_top_k (object): Invalid result limit supplied by pytest.

    Returns:
        None: Passes when ValidationError is raised.
    """

    with pytest.raises(ValidationError):
        fake_repo.search_chunks(_vector(0), top_k=invalid_top_k)


def test_invalid_ingestion_does_not_modify_repository(
    fake_repo: InMemoryRepository,
) -> None:
    """
    Verify that invalid chunk ordering cannot partially insert an article.

    Args:
        fake_repo (InMemoryRepository): Empty repository fixture.

    Returns:
        None: Passes when invalid ingestion leaves storage unchanged.
    """

    article = _article(
        1,
        "Alpha",
        topic="physics",
        categories=(),
        length_chars=100,
        revised_at=datetime(2026, 5, 1),
    )

    # Chunk indices must begin at zero.
    invalid_chunk = ChunkRecord(
        chunk_index=1,
        text="Invalid chunk ordering",
        embedding=_vector(0),
    )

    with pytest.raises(ValidationError):
        fake_repo.ingest_article(article, [invalid_chunk])

    assert fake_repo.count_articles() == 0
    assert fake_repo.count_chunks() == 0


def test_fake_matches_titles_and_links_exactly(
    fake_repo: InMemoryRepository,
) -> None:
    """Keep case-sensitive title matching consistent with MariaDB.

    Args:
        fake_repo: Empty in-memory repository.
    """
    first = _article(
        1,
        "Apple",
        topic="testing",
        categories=(),
        length_chars=100,
        revised_at=datetime(2026, 1, 1),
    )
    second = _article(
        2,
        "APPLE",
        topic="testing",
        categories=(),
        length_chars=100,
        revised_at=datetime(2026, 1, 1),
    )
    source = _article(
        3,
        "Source",
        topic="testing",
        categories=(),
        length_chars=100,
        revised_at=datetime(2026, 1, 1),
        linked_titles=("Apple", "APPLE"),
    )

    for article in (first, second, source):
        chunk = ChunkRecord(
            chunk_index=0,
            text="Example content",
            embedding=_vector(0),
        )
        assert fake_repo.ingest_article(article, [chunk]) == "inserted"

    assert fake_repo.get_article_by_title("Apple") == first
    assert fake_repo.get_article_by_title("APPLE") == second
    assert fake_repo.get_article_by_title("apple") is None

    assert fake_repo.resolve_links() == 2
    assert {article.article_id for article in fake_repo.get_linked_articles(source.article_id)} == {
        1,
        2,
    }
