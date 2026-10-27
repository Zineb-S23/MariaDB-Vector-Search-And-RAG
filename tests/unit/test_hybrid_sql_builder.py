"""Test safe SQL generation for database-native hybrid search."""

from datetime import datetime

from wikirag.db.queries import build_hybrid_search_query
from wikirag.models import SearchFilters


def test_hybrid_query_includes_every_filter() -> None:
    """Verify every shared filter produces the required SQL predicate."""
    edited_since = datetime(2026, 1, 1)

    filters = SearchFilters(
        topics=("Physics", "Biology"),
        categories=("Science",),
        article_ids=(7, 8),
        exclude_article_ids=(9,),
        min_length_chars=500,
        edited_since=edited_since,
        linked_from_article_id=10,
        max_distance=0.3,
    )

    statement, params = build_hybrid_search_query(
        filters,
        vector_json="[1.0, 0.0]",
        top_k=5,
    )

    assert "LOWER(a.topic) IN (%s, %s)" in statement
    assert "LOWER(cat.name) IN (%s)" in statement
    assert "a.article_id IN (%s, %s)" in statement
    assert "a.article_id NOT IN (%s)" in statement
    assert "a.length_chars > %s" in statement
    assert "a.revised_at >= %s" in statement
    assert "l.source_article_id = %s" in statement
    assert "ranked.distance <= %s" in statement

    assert params == (
        "[1.0, 0.0]",
        "physics",
        "biology",
        "science",
        7,
        8,
        9,
        500,
        edited_since,
        10,
        0.3,
        5,
    )

    assert statement.count("%s") == len(params)


def test_hybrid_query_preserves_exact_ranking() -> None:
    """Ensure filters are applied before exact top-K ranking."""
    statement, params = build_hybrid_search_query(
        SearchFilters(categories=("Science",)),
        vector_json="[1.0]",
        top_k=3,
    )

    assert "IGNORE INDEX (embedding)" in statement
    assert "VEC_DISTANCE_COSINE" in statement
    assert "FROM article_categories AS ac" in statement
    assert "ORDER BY" in statement
    assert "LIMIT %s" in statement

    assert params == ("[1.0]", "science", 3)
