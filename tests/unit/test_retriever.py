"""Unit tests for semantic and hybrid retrieval."""

from dataclasses import FrozenInstanceError

import pytest

from wikirag.errors import ValidationError
from wikirag.models import SearchFilters
from wikirag.retrieval.retriever import Retriever


def test_semantic_retrieval_returns_result(
    fake_repo,
    fake_embedder,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)

    result = retriever.retrieve("What is MariaDB?")

    assert result.query == "What is MariaDB?"
    assert result.mode == "semantic"
    assert result.top_k == 5
    assert result.filters is None
    assert result.hits == ()
    assert result.embed_ms >= 0
    assert result.search_ms >= 0
    assert result.latency_ms >= 0


def test_semantic_mode_ignores_filters(
    fake_repo,
    fake_embedder,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)
    filters = SearchFilters(topics=("physics",))

    result = retriever.retrieve(
        "What is MariaDB?",
        mode="semantic",
        filters=filters,
    )

    assert result.filters is None


def test_hybrid_mode_requires_active_filters(
    fake_repo,
    fake_embedder,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)

    with pytest.raises(ValidationError):
        retriever.retrieve("What is MariaDB?", mode="hybrid")

    with pytest.raises(ValidationError):
        retriever.retrieve(
            "What is MariaDB?",
            mode="hybrid",
            filters=SearchFilters(),
        )


def test_hybrid_mode_passes_filters_to_result(
    fake_repo,
    fake_embedder,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)
    filters = SearchFilters(topics=("physics",))

    result = retriever.retrieve(
        "What is MariaDB?",
        mode="hybrid",
        filters=filters,
    )

    assert result.mode == "hybrid"
    assert result.filters == filters


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_rejects_invalid_query(
    fake_repo,
    fake_embedder,
    query,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)

    with pytest.raises(ValidationError):
        retriever.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, 101, True, 1.5, "5", None])
def test_rejects_invalid_top_k(
    fake_repo,
    fake_embedder,
    top_k,
) -> None:
    retriever = Retriever(fake_repo, fake_embedder)

    with pytest.raises(ValidationError):
        retriever.retrieve("MariaDB", top_k=top_k)


def test_rejects_unknown_mode(fake_repo, fake_embedder) -> None:
    retriever = Retriever(fake_repo, fake_embedder)

    with pytest.raises(ValidationError):
        retriever.retrieve("MariaDB", mode="unknown")


def test_result_is_immutable(fake_repo, fake_embedder) -> None:
    retriever = Retriever(fake_repo, fake_embedder)
    result = retriever.retrieve("MariaDB")

    with pytest.raises(FrozenInstanceError):
        result.query = "changed"


def test_retrieval_preserves_search_hits(
    fake_repo,
    fake_embedder,
) -> None:
    from datetime import datetime

    from wikirag.config import EMBEDDING_DIM
    from wikirag.models import ArticleRecord, ChunkRecord

    vector = tuple(1.0 if index == 0 else 0.0 for index in range(EMBEDDING_DIM))
    article = ArticleRecord(
        article_id=1,
        title="MariaDB",
        url="https://en.wikipedia.org/wiki/MariaDB",
        revision_id=1,
        fingerprint="a" * 64,
        revised_at=datetime(2026, 1, 1),
        length_chars=100,
        fetched_at=datetime(2026, 1, 1),
    )
    fake_repo.ingest_article(
        article,
        [ChunkRecord(chunk_index=0, text="MariaDB is a database.", embedding=vector)],
    )

    # The fake embedder's query vector is deterministic; the repository
    # should return the stored chunk as a typed SearchHit.
    retriever = Retriever(fake_repo, fake_embedder)
    result = retriever.retrieve("MariaDB")

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.article_id == 1
    assert hit.article_title == "MariaDB"
    assert hit.text == "MariaDB is a database."
    assert 0.0 <= hit.distance <= 2.0
    assert hit.score == pytest.approx(1.0 - hit.distance)
