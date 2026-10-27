"""Inspect MariaDB native vector-search execution plans."""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime
from typing import Literal
from uuid import uuid4

import pymysql
from pymysql.connections import Connection

from wikirag.config import DbConfig, load_settings
from wikirag.db.connection import get_connection
from wikirag.db.queries import (
    COUNT_VISIBILITY_ARTICLES,
    COUNT_VISIBILITY_CHUNKS,
    DELETE_ARTICLE_BY_ID,
    INSERT_ARTICLE,
    INSERT_CHUNK,
    SEARCH_CHUNKS_SEMANTIC,
    TRANSACTION_ISOLATION_SQL,
    build_hybrid_search_query,
)
from wikirag.db.repository import WikiRepository
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    ValidationError,
)
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


def _insert_visibility_rows(
    connection: Connection,
    *,
    article_id: int,
    identifier: str,
    vector_json: str,
) -> None:
    """Insert article metadata and a native vector without committing.

    Args:
        connection: Writer connection with an active transaction.
        article_id: Unique synthetic article identifier.
        identifier: Unique identifier used for synthetic metadata.
        vector_json: Valid serialized 384-dimensional embedding.

    Returns:
        None: Article and chunk are inserted into the active transaction.

    Raises:
        pymysql.MySQLError: If an insertion fails.
    """
    timestamp = datetime(2026, 1, 1)
    content = "Transaction visibility test chunk"

    with connection.cursor() as cursor:
        cursor.execute(
            INSERT_ARTICLE,
            (
                article_id,
                f"WikiRAG visibility test {identifier}",
                f"https://example.invalid/wikirag/{identifier}",
                1,
                timestamp,
                len(content),
                identifier * 2,
                "experiment",
                "en",
                timestamp,
            ),
        )

        cursor.execute(
            INSERT_CHUNK,
            (
                article_id,
                0,
                None,
                content,
                len(content),
                vector_json,
            ),
        )


def _observe_visibility(
    reader: Connection,
    *,
    article_id: int,
    vector_json: str,
) -> dict[str, int]:
    """Count visible metadata, vectors, and native search results.

    Each observation ends its reader transaction so the following
    observation uses a fresh snapshot under both isolation levels.

    Args:
        reader: Independent MariaDB reader connection.
        article_id: Synthetic article being observed.
        vector_json: Query vector matching the inserted chunk.

    Returns:
        dict[str, int]: Counts of visible articles, chunks, and
            matching native vector-search results.

    Raises:
        pymysql.MySQLError: If a database query fails.
    """
    with reader.cursor() as cursor:
        cursor.execute(
            COUNT_VISIBILITY_ARTICLES,
            (article_id,),
        )
        articles = int(cursor.fetchone()["total"])

        cursor.execute(
            COUNT_VISIBILITY_CHUNKS,
            (article_id,),
        )
        chunks = int(cursor.fetchone()["total"])

        # Use the application's native indexed semantic-search query.
        cursor.execute(
            SEARCH_CHUNKS_SEMANTIC,
            (vector_json, vector_json, 10),
        )

        hits = sum(row["article_id"] == article_id for row in cursor.fetchall())

        # Compare HNSW retrieval against an exact vector scan.
        cursor.execute(
            """
            SELECT c.article_id
            FROM chunks AS c IGNORE INDEX (embedding)
            ORDER BY VEC_DISTANCE_COSINE(
                c.embedding,
                VEC_FromText(%s)
            ) ASC
            LIMIT 10
            """,
            (vector_json,),
        )

        exact_hits = sum(row["article_id"] == article_id for row in cursor.fetchall())

    # End the read transaction to avoid retaining an old snapshot.
    reader.commit()

    return {
        "articles": articles,
        "chunks": chunks,
        "q1_hits": hits,
        "exact_hits": exact_hits,
    }


def demo_transaction_visibility(
    config: DbConfig | None = None,
    *,
    isolation_levels: tuple[str, ...] = (
        "REPEATABLE READ",
        "READ COMMITTED",
    ),
) -> dict[str, object]:
    """Demonstrate transactional visibility of native vectors and metadata.

    Test independent writer and reader connections using the requested
    transaction isolation levels. Observe uncommitted inserts, rollback,
    commit, fresh-reader visibility, and final cleanup.

    Args:
        config: Dedicated MariaDB test database configuration.
        isolation_levels: Isolation levels to investigate in order.
            Defaults to REPEATABLE READ and READ COMMITTED.

    Returns:
        dict[str, object]: Visibility observations for each isolation
            level, including article counts, chunk counts, indexed
            vector hits, and exact vector hits.

    Raises:
        ValidationError: If the database is unsafe or an isolation
            level is unsupported.
        DatabaseConnectionError: If MariaDB becomes unavailable.
        DatabaseError: If a database query or transaction fails.
    """
    settings = load_settings()
    selected_config = config if config is not None else settings.db

    # Never run destructive experiment operations on the application DB.
    if (
        not selected_config.database.endswith("_test")
        or selected_config.database == settings.db.database
    ):
        raise ValidationError("Transaction visibility requires a separate *_test database")

    # Only explicitly supported isolation levels are allowed.
    if (
        not isinstance(isolation_levels, tuple)
        or not isolation_levels
        or any(
            not isinstance(level, str) or level not in TRANSACTION_ISOLATION_SQL
            for level in isolation_levels
        )
    ):
        raise ValidationError("Unsupported transaction isolation level")

    vector_json = json.dumps([1.0] + [0.0] * 383)

    observations: dict[str, dict[str, dict[str, int]]] = {}

    try:
        for isolation in isolation_levels:
            statement = TRANSACTION_ISOLATION_SQL[isolation]

            unique = uuid4()
            article_id = (unique.int & ((1 << 63) - 1)) or 1

            # Writer and reader use independent MariaDB connections.
            with (
                closing(get_connection(selected_config)) as writer,
                closing(get_connection(selected_config)) as reader,
            ):
                with writer.cursor() as cursor:
                    cursor.execute(statement)

                with reader.cursor() as cursor:
                    cursor.execute(statement)

                states: dict[str, dict[str, int]] = {}

                try:
                    # Observe database state before any insertion.
                    states["before"] = _observe_visibility(
                        reader,
                        article_id=article_id,
                        vector_json=vector_json,
                    )

                    # Insert article metadata and vector without committing.
                    writer.begin()

                    _insert_visibility_rows(
                        writer,
                        article_id=article_id,
                        identifier=unique.hex,
                        vector_json=vector_json,
                    )

                    # The independent reader must not see uncommitted rows.
                    states["uncommitted"] = _observe_visibility(
                        reader,
                        article_id=article_id,
                        vector_json=vector_json,
                    )

                    # Roll back the first insertion.
                    writer.rollback()

                    states["after_rollback"] = _observe_visibility(
                        reader,
                        article_id=article_id,
                        vector_json=vector_json,
                    )

                    # Repeat the insertion using a new transaction.
                    writer.begin()

                    _insert_visibility_rows(
                        writer,
                        article_id=article_id,
                        identifier=unique.hex,
                        vector_json=vector_json,
                    )

                    states["before_commit"] = _observe_visibility(
                        reader,
                        article_id=article_id,
                        vector_json=vector_json,
                    )

                    # Make the new article and its vector permanent.
                    writer.commit()

                    states["after_commit"] = _observe_visibility(
                        reader,
                        article_id=article_id,
                        vector_json=vector_json,
                    )

                    # Check whether a brand-new connection sees the vector.
                    with closing(get_connection(selected_config)) as fresh_reader:
                        with fresh_reader.cursor() as cursor:
                            cursor.execute(statement)

                        states["after_commit_fresh_reader"] = _observe_visibility(
                            fresh_reader,
                            article_id=article_id,
                            vector_json=vector_json,
                        )

                finally:
                    # Roll back any incomplete transaction before cleanup.
                    writer.rollback()
                    reader.rollback()

                    # Remove only the synthetic article from this run.
                    # Foreign-key cascading also removes its chunks.

                    with writer.cursor() as cursor:
                        # Remove indexed vectors directly before deleting their parent.
                        cursor.execute(
                            "DELETE FROM chunks WHERE article_id = %s",
                            (article_id,),
                        )

                        cursor.execute(
                            DELETE_ARTICLE_BY_ID,
                            (article_id,),
                        )

                    writer.commit()

                states["after_cleanup"] = _observe_visibility(
                    reader,
                    article_id=article_id,
                    vector_json=vector_json,
                )

                observations[isolation] = states

    except pymysql.err.OperationalError:
        raise DatabaseConnectionError(
            "MariaDB connection failed during visibility experiment"
        ) from None

    except pymysql.MySQLError:
        raise DatabaseError("MariaDB transaction visibility experiment failed") from None

    return {
        "experiment": "transaction_visibility",
        "database": selected_config.database,
        "isolation_levels": observations,
        "notes": (
            "Each reader observation ends its transaction to obtain "
            "a fresh snapshot. Synthetic rows are removed after "
            "each experiment. Both indexed and exact vector-search "
            "visibility are recorded."
        ),
    }
