"""Check shared behavior of the fake and MariaDB repositories."""

from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from datetime import datetime

import pytest
from fakes import InMemoryRepository

from wikirag.config import EMBEDDING_DIM, load_settings
from wikirag.db.connection import get_connection
from wikirag.db.repository import WikiRepository
from wikirag.db.schema import init_schema
from wikirag.errors import EmbeddingDimensionError, ValidationError
from wikirag.models import ArticleRecord, ChunkRecord, SearchFilters


@pytest.fixture(params=("fake", "mariadb"))
def repository(
    request: pytest.FixtureRequest,
) -> Iterator[InMemoryRepository | WikiRepository]:
    """Provide either repository with isolated storage.

    Args:
        request: Pytest request containing the repository variant.

    Yields:
        An empty fake repository or an initialized MariaDB repository.

    Raises:
        ValidationError: If the configured test database is unsafe.
    """
    if request.param == "fake":
        repo = InMemoryRepository()
    else:
        settings = load_settings()

        if (
            not settings.db_name_test.endswith("_test")
            or settings.db_name_test == settings.db.database
        ):
            raise ValidationError("Parity tests require an isolated *_test database")

        config = replace(settings.db, database=settings.db_name_test)

        # Recreate only the dedicated test schema.
        with closing(get_connection(config)) as conn:
            init_schema(conn, recreate=True)

        repo = WikiRepository(config)

    try:
        yield repo
    finally:
        repo.close()


def _article(
    article_id: int,
    title: str,
    *,
    fingerprint: str = "a" * 64,
    linked_titles: tuple[str, ...] = (),
) -> ArticleRecord:
    """Create deterministic article metadata for contract tests.

    Args:
        article_id: Unique Wikipedia-style article identifier.
        title: Case-sensitive article title.
        fingerprint: SHA-256-style fingerprint.
        linked_titles: Outgoing Wikipedia link titles.

    Returns:
        Valid article metadata for either repository.
    """
    return ArticleRecord(
        article_id=article_id,
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title}",
        revision_id=1,
        fingerprint=fingerprint,
        revised_at=datetime(2026, 1, 1),
        fetched_at=datetime(2026, 1, 2),
        length_chars=100,
        topic="science",
        categories=("Science",),
        linked_titles=linked_titles,
    )


def _chunk() -> ChunkRecord:
    """Create one valid 384-dimensional chunk.

    Returns:
        ChunkRecord: A chunk with a nonzero embedding.
    """
    return ChunkRecord(
        chunk_index=0,
        text="Example scientific content.",
        embedding=(1.0,) + (0.0,) * (EMBEDDING_DIM - 1),
    )


def test_insert_skip_update_parity(
    repository: InMemoryRepository | WikiRepository,
) -> None:
    """Verify identical insertion, skipping, and updating outcomes.

    Args:
        repository: Isolated fake or MariaDB repository.
    """
    article = _article(920001, "Physics")

    assert repository.ingest_article(article, (_chunk(),)) == "inserted"
    assert repository.count_articles() == 1
    assert repository.count_chunks() == 1
    assert repository.get_fingerprint(article.article_id) == "a" * 64

    assert repository.ingest_article(article, (_chunk(),)) == "skipped"

    updated = replace(
        article,
        fingerprint="b" * 64,
        revision_id=2,
    )

    assert repository.ingest_article(updated, (_chunk(),)) == "updated"
    assert repository.count_articles() == 1
    assert repository.count_chunks() == 1
    assert repository.get_fingerprint(article.article_id) == "b" * 64

    stored = repository.get_article(article.article_id)
    assert stored is not None
    assert stored.revision_id == 2


def test_titles_links_and_deletion_parity(
    repository: InMemoryRepository | WikiRepository,
) -> None:
    """Verify exact titles, resolved links, and deletion behavior.

    Args:
        repository: Isolated fake or MariaDB repository.
    """
    first = _article(920001, "Apple")
    second = _article(920002, "APPLE")
    source = _article(
        920003,
        "Source",
        linked_titles=("Apple", "APPLE", "Outside"),
    )

    for article in (first, second, source):
        assert repository.ingest_article(article, (_chunk(),)) == "inserted"

    assert repository.get_article_by_title("Apple").article_id == 920001
    assert repository.get_article_by_title("APPLE").article_id == 920002
    assert repository.get_article_by_title("apple") is None

    assert repository.resolve_links() == 2

    linked = repository.get_linked_articles(source.article_id)
    assert [article.article_id for article in linked] == [920001, 920002]

    assert repository.delete_article(920001) is True
    assert repository.delete_article(920001) is False

    assert repository.get_article(920001) is None
    assert repository.count_articles() == 2
    assert repository.count_chunks() == 2

    remaining = repository.get_linked_articles(source.article_id)
    assert [article.article_id for article in remaining] == [920002]


def test_invalid_embedding_preserves_state(
    repository: InMemoryRepository | WikiRepository,
) -> None:
    """Reject incorrect dimensions without modifying stored data.

    Args:
        repository: Isolated fake or MariaDB repository.
    """
    article = _article(920001, "Physics")

    assert repository.ingest_article(article, (_chunk(),)) == "inserted"

    invalid_chunk = replace(
        _chunk(),
        embedding=(1.0,) * (EMBEDDING_DIM - 1),
    )

    updated = replace(article, fingerprint="c" * 64)

    with pytest.raises(EmbeddingDimensionError):
        repository.ingest_article(updated, (invalid_chunk,))

    assert repository.get_fingerprint(article.article_id) == "a" * 64
    assert repository.count_articles() == 1
    assert repository.count_chunks() == 1


@pytest.mark.parametrize(
    ("filters", "expected_article_ids"),
    (
        pytest.param(
            SearchFilters(topics=("BIOLOGY",)),
            (920002,),
            id="topic",
        ),
        pytest.param(
            SearchFilters(categories=("life",)),
            (920002,),
            id="category",
        ),
        pytest.param(
            SearchFilters(article_ids=(920002,)),
            (920002,),
            id="included-article",
        ),
        pytest.param(
            SearchFilters(exclude_article_ids=(920001,)),
            (920002,),
            id="excluded-article",
        ),
        pytest.param(
            SearchFilters(min_length_chars=100),
            (920002,),
            id="article-length",
        ),
        pytest.param(
            SearchFilters(edited_since=datetime(2026, 2, 1)),
            (920002,),
            id="revision-date",
        ),
        pytest.param(
            SearchFilters(linked_from_article_id=920001),
            (920002,),
            id="article-links",
        ),
        pytest.param(
            SearchFilters(max_distance=0.0),
            (920001, 920002),
            id="distance-boundary",
        ),
        pytest.param(
            SearchFilters(
                topics=("missing", "biology"),
                categories=("missing", "Life"),
            ),
            (920002,),
            id="or-within-fields",
        ),
        pytest.param(
            SearchFilters(
                topics=("BIOLOGY",),
                categories=("life",),
                min_length_chars=100,
                linked_from_article_id=920001,
            ),
            (920002,),
            id="combined-filters",
        ),
        pytest.param(
            SearchFilters(
                topics=("biology",),
                categories=("Science",),
            ),
            (),
            id="conflicting-filters",
        ),
    ),
)
def test_filtered_search_parity(
    repository: InMemoryRepository | WikiRepository,
    filters: SearchFilters,
    expected_article_ids: tuple[int, ...],
) -> None:
    """Verify identical filtered search results across repositories.

    Args:
        repository: Isolated fake or real MariaDB repository.
        filters: Relational and distance constraints to test.
        expected_article_ids: Expected ranked article identifiers.
    """
    source = _article(
        920001,
        "Source",
        linked_titles=("Biology",),
    )

    biology = replace(
        _article(
            920002,
            "Biology",
            fingerprint="b" * 64,
        ),
        topic="biology",
        categories=("Life",),
        length_chars=150,
        revised_at=datetime(2026, 4, 1),
    )

    # Identical embeddings make the deterministic tie-breakers testable.
    for article in (source, biology):
        assert (
            repository.ingest_article(
                article,
                (_chunk(),),
            )
            == "inserted"
        )

    assert repository.resolve_links() == 1

    query_vector = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    hits = repository.search_chunks(
        query_vector,
        top_k=5,
        filters=filters,
    )

    assert tuple(hit.article_id for hit in hits) == expected_article_ids

    for hit in hits:
        assert hit.score == pytest.approx(1.0 - hit.distance)
        assert 0.0 <= hit.distance <= 2.0
