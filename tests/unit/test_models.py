"""
Unit tests for shared WikiRAG data models.

These tests protect the field names, immutable dataclass contracts,
and search-filter behavior agreed upon by the three teammates.

They do not require a running MariaDB server or an embedding model.
"""

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime

import pytest

from wikirag.errors import ValidationError
from wikirag.models import (
    ArticleRecord,
    ChunkRecord,
    ContextBundle,
    EvalQuery,
    EvalReport,
    IngestReport,
    ModeMetrics,
    RAGAnswer,
    RetrievalResult,
    SearchFilters,
    SearchHit,
    StoredChunk,
)


def test_shared_model_field_names() -> None:
    """
    Verify that every shared model exposes the agreed field names.

    These names form the interface between ingestion, database,
    retrieval, RAG, and evaluation.

    Returns:
        None: The test passes when every field list matches.

    Raises:
        AssertionError: If a model's public fields change.
    """

    expected_fields = {
        ArticleRecord: (
            "article_id",
            "title",
            "url",
            "revision_id",
            "fingerprint",
            "revised_at",
            "length_chars",
            "fetched_at",
            "topic",
            "language",
            "categories",
            "linked_titles",
        ),
        ChunkRecord: (
            "chunk_index",
            "text",
            "embedding",
            "section_title",
        ),
        StoredChunk: (
            "chunk_id",
            "article_id",
            "article_title",
            "chunk_index",
            "section_title",
            "text",
        ),
        SearchFilters: (
            "topics",
            "categories",
            "article_ids",
            "exclude_article_ids",
            "min_length_chars",
            "edited_since",
            "linked_from_article_id",
            "max_distance",
        ),
        SearchHit: (
            "chunk_id",
            "article_id",
            "article_title",
            "article_url",
            "chunk_index",
            "section_title",
            "text",
            "distance",
            "score",
        ),
        IngestReport: (
            "inserted",
            "updated",
            "skipped",
            "failed",
        ),
        RetrievalResult: (
            "query",
            "mode",
            "top_k",
            "filters",
            "hits",
            "embed_ms",
            "search_ms",
            "latency_ms",
        ),
        ContextBundle: (
            "text",
            "sources",
            "truncated",
        ),
        RAGAnswer: (
            "question",
            "answer",
            "context",
            "retrieval",
            "generator_name",
            "latency_ms",
        ),
        EvalQuery: (
            "query_id",
            "text",
            "relevant_article_ids",
            "filters",
            "notes",
        ),
        ModeMetrics: (
            "n_queries",
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "latency_ms_p50",
            "latency_ms_p95",
        ),
        EvalReport: (
            "embedding_model",
            "corpus_articles",
            "corpus_chunks",
            "ks",
            "modes",
            "per_query",
        ),
    }

    # Compare field names in order, since dataclass constructors use that order.
    for model_class, expected in expected_fields.items():
        actual = tuple(field.name for field in fields(model_class))

        assert actual == expected, f"{model_class.__name__}: expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "model_class",
    [
        ArticleRecord,
        ChunkRecord,
        StoredChunk,
        SearchFilters,
        SearchHit,
        IngestReport,
        RetrievalResult,
        ContextBundle,
        RAGAnswer,
        EvalQuery,
        ModeMetrics,
        EvalReport,
    ],
)
def test_shared_models_are_frozen_and_slotted(model_class: type) -> None:
    """
    Verify that every shared dataclass is frozen and uses slots.

    Args:
        model_class (type): Dataclass selected by pytest.

    Returns:
        None: The test passes when both requirements are satisfied.

    Raises:
        AssertionError: If the class is not frozen or lacks slots.
    """

    assert model_class.__dataclass_params__.frozen
    assert hasattr(model_class, "__slots__")


def test_frozen_model_rejects_field_reassignment() -> None:
    """
    Verify that a shared model rejects field reassignment.

    Returns:
        None: The test passes when reassignment raises the expected
        exception.

    Raises:
        AssertionError: If reassignment does not raise the exception.
    """

    chunk = ChunkRecord(
        chunk_index=0,
        text="Example Wikipedia chunk",
        embedding=(0.0,) * 384,
    )

    with pytest.raises(FrozenInstanceError):
        chunk.text = "Changed text"


def test_empty_search_filters() -> None:
    """
    Verify that a default SearchFilters object has no active filters.

    Returns:
        None: The test passes when is_empty() returns True.

    Raises:
        AssertionError: If default filters are incorrectly active.
    """

    filters = SearchFilters()

    assert filters.is_empty() is True


@pytest.mark.parametrize(
    "filters",
    [
        SearchFilters(topics=("Physics",)),
        SearchFilters(categories=("Science",)),
        SearchFilters(article_ids=(123,)),
        SearchFilters(exclude_article_ids=(456,)),
        SearchFilters(min_length_chars=0),
        SearchFilters(edited_since=datetime(2026, 1, 1)),
        SearchFilters(linked_from_article_id=123),
        SearchFilters(max_distance=0.0),
    ],
)
def test_each_search_filter_is_detected(filters: SearchFilters) -> None:
    """
    Verify that each supported filter counts as active.

    Zero is an intentional value for numeric filters and must not
    be mistaken for an unset filter.

    Args:
        filters (SearchFilters): Filter configuration supplied by pytest.

    Returns:
        None: The test passes when is_empty() returns False.

    Raises:
        AssertionError: If an active filter is incorrectly ignored.
    """

    assert filters.is_empty() is False


def test_article_record_defaults() -> None:
    """
    Verify the documented defaults for optional article metadata.

    Returns:
        None: The test passes when optional fields have the correct
        default values.

    Raises:
        AssertionError: If an article default changes unexpectedly.
    """

    article = ArticleRecord(
        article_id=123,
        title="Computer science",
        url="https://en.wikipedia.org/wiki/Computer_science",
        revision_id=456,
        fingerprint="a" * 64,
        revised_at=datetime(2026, 1, 1),
        length_chars=5000,
        fetched_at=datetime(2026, 1, 2),
    )

    assert article.topic is None
    assert article.language == "en"
    assert article.categories == ()
    assert article.linked_titles == ()


def test_evaluation_query_preserves_relevant_articles() -> None:
    """
    Verify that evaluation labels use article-level identifiers.

    Returns:
        None: The test passes when the labels and defaults match
        the shared evaluation contract.

    Raises:
        AssertionError: If evaluation fields are stored incorrectly.
    """

    query = EvalQuery(
        query_id="q001",
        text="What is a relational database?",
        relevant_article_ids=frozenset({123, 456}),
    )

    assert query.relevant_article_ids == frozenset({123, 456})
    assert query.filters is None
    assert query.notes == ""


@pytest.mark.parametrize(
    "filters",
    (
        SearchFilters(topics="physics"),
        SearchFilters(topics=("physics", "")),
        SearchFilters(categories=(123,)),
        SearchFilters(categories=["Science"]),
        SearchFilters(article_ids=(True,)),
        SearchFilters(article_ids=(-1,)),
        SearchFilters(exclude_article_ids=("7",)),
        SearchFilters(min_length_chars=-1),
        SearchFilters(min_length_chars=True),
        SearchFilters(edited_since="2026-01-01"),
        SearchFilters(edited_since=datetime(2026, 1, 1, tzinfo=UTC)),
        SearchFilters(linked_from_article_id=0),
        SearchFilters(max_distance=float("nan")),
        SearchFilters(max_distance="0.3"),
    ),
)
def test_search_filters_reject_invalid_values(
    filters: SearchFilters,
) -> None:
    """Reject malformed values from every shared filter category.

    Args:
        filters: Search constraints containing an invalid value.
    """
    with pytest.raises(ValidationError):
        filters.validate()


def test_search_filters_accept_valid_boundaries() -> None:
    """Accept valid empty and boundary-valued search constraints."""
    SearchFilters().validate()

    filters = SearchFilters(
        topics=("Physics",),
        categories=("Science",),
        article_ids=(1,),
        exclude_article_ids=(2,),
        min_length_chars=0,
        edited_since=datetime(2026, 1, 1),
        linked_from_article_id=3,
        max_distance=0.0,
    )

    filters.validate()
