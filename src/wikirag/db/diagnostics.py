"""Inspect MariaDB native vector-search execution plans."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from typing import Literal

from wikirag.config import DbConfig, load_settings
from wikirag.db.connection import get_connection
from wikirag.db.queries import (
    SEARCH_CHUNKS_SEMANTIC,
    build_hybrid_search_query,
)
from wikirag.db.repository import WikiRepository
from wikirag.errors import ValidationError
from wikirag.models import SearchFilters


def explain_search(
    query_embedding: Sequence[float],
    *,
    top_k: int = 10,
    filters: SearchFilters | None = None,
    distance: Literal["cosine", "euclidean"] = "cosine",
    config: DbConfig | None = None,
) -> list[dict[str, object]]:
    """Inspect MariaDB's execution plan for native vector retrieval.

    Unfiltered searches preserve the semantic query's structure.
    Filtered searches use exact ranking with relational predicates.

    Args:
        query_embedding: Query embedding containing 384 numeric values.
        top_k: Maximum number of results, between 1 and 100.
        filters: Optional validated relational search constraints.
        distance: Distance function to inspect.
        config: Optional database configuration. Defaults to the
            configured application database.

    Returns:
        list[dict[str, object]]: MariaDB EXPLAIN rows describing
            the chosen query execution plan.

    Raises:
        ValidationError: If the limit, metric, or filters are invalid.
        EmbeddingDimensionError: If the query vector is invalid.
        DatabaseConnectionError: If MariaDB cannot be reached.
        DatabaseError: If the database rejects the query.
    """
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValidationError("top_k must be an integer between 1 and 100")

    if distance not in ("cosine", "euclidean"):
        raise ValidationError("distance must be cosine or euclidean")

    if filters is not None:
        if not isinstance(filters, SearchFilters):
            raise ValidationError("filters must be SearchFilters or None")

        filters.validate()

    # Reuse the repository's vector validation and float32 serialization.
    vector_json = WikiRepository._serialize_search_vector(query_embedding)

    if filters is not None and not filters.is_empty():
        statement, params = build_hybrid_search_query(
            filters,
            vector_json=vector_json,
            top_k=top_k,
        )
    else:
        statement = SEARCH_CHUNKS_SEMANTIC
        params = (vector_json, vector_json, top_k)

    if distance == "euclidean":
        # The metric name is selected from two fixed trusted choices.
        statement = statement.replace(
            "VEC_DISTANCE_COSINE",
            "VEC_DISTANCE_EUCLIDEAN",
        )

    selected_config = config if config is not None else load_settings().db

    with closing(get_connection(selected_config)) as conn, conn.cursor() as cursor:
        cursor.execute("EXPLAIN " + statement, params)
        return [dict(row) for row in cursor.fetchall()]
