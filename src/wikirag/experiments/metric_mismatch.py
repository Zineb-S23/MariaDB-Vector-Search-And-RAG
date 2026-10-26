"""Experiment C: compare vector-distance expressions and index plans."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from statistics import mean, median, quantiles, stdev
from time import perf_counter
from typing import TYPE_CHECKING

from wikirag.config import DbConfig, load_settings
from wikirag.db.connection import get_connection
from wikirag.db.repository import WikiRepository
from wikirag.errors import ValidationError

if TYPE_CHECKING:
    from wikirag.embeddings import Embedder


# Use fixed SQL statements to compare the same query structure.
SQL_VARIANTS = {
    "cosine": """
        SELECT c.chunk_id
        FROM chunks AS c
        ORDER BY VEC_DISTANCE_COSINE(
            c.embedding, VEC_FromText(%s)
        ) ASC
        LIMIT %s
    """,
    "euclidean": """
        SELECT c.chunk_id
        FROM chunks AS c
        ORDER BY VEC_DISTANCE_EUCLIDEAN(
            c.embedding, VEC_FromText(%s)
        ) ASC
        LIMIT %s
    """,
    "wrapped_cosine": """
        SELECT c.chunk_id
        FROM chunks AS c
        ORDER BY (
            1 - VEC_DISTANCE_COSINE(
                c.embedding, VEC_FromText(%s)
            )
        ) DESC
        LIMIT %s
    """,
}


def run_metric_mismatch(
    embedder: Embedder,
    *,
    sample_query: str,
    repeats: int = 5,
    config: DbConfig | None = None,
) -> dict[str, object]:
    """Measure how distance expressions affect MariaDB vector retrieval.

    Each query receives one warm-up execution followed by repeated
    measurements. EXPLAIN is collected for the exact SQL being timed.

    Args:
        embedder: Shared embedding implementation for the query.
        sample_query: Non-empty question used to generate a query vector.
        repeats: Number of timed executions, at least five.
        config: MariaDB connection settings; defaults to the app database.

    Returns:
        dict[str, object]: Database version, execution plans, index usage,
            latency statistics, and ratios relative to cosine retrieval.

    Raises:
        ValidationError: If arguments are invalid or no chunks are stored.
        EmbeddingDimensionError: If the embedder returns an invalid vector.
        DatabaseConnectionError: If MariaDB is unavailable.
        DatabaseError: If MariaDB rejects a query.
    """
    if not isinstance(sample_query, str) or not sample_query.strip():
        raise ValidationError("sample_query must be a non-empty string")

    if type(repeats) is not int or repeats < 5:
        raise ValidationError("repeats must be an integer of at least 5")

    selected_config = config if config is not None else load_settings().db

    query_embedding: Sequence[float] = embedder.embed_query(sample_query)

    # Apply exactly the same vector validation as normal retrieval.
    vector_json = WikiRepository._serialize_search_vector(query_embedding)

    variants: dict[str, dict[str, object]] = {}

    with closing(get_connection(selected_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT VERSION() AS version")
        server_version = str(cursor.fetchone()["version"])

        cursor.execute("SELECT COUNT(*) AS total FROM chunks")
        chunk_count = int(cursor.fetchone()["total"])

        if chunk_count == 0:
            raise ValidationError("Metric-mismatch experiment requires stored chunks")

        top_k = min(10, chunk_count)
        params = (vector_json, top_k)

        for name, statement in SQL_VARIANTS.items():
            cursor.execute("EXPLAIN " + statement, params)
            plan = [dict(row) for row in cursor.fetchall()]

            uses_vector_index = any(
                row.get("table") == "c" and row.get("key") == "embedding" for row in plan
            )

            # Warm the query before collecting latency samples.
            cursor.execute(statement, params)
            cursor.fetchall()

            durations_ms: list[float] = []
            rows_returned = 0

            for _ in range(repeats):
                started = perf_counter()

                cursor.execute(statement, params)
                rows = cursor.fetchall()

                elapsed_ms = (perf_counter() - started) * 1000.0

                durations_ms.append(elapsed_ms)
                rows_returned = len(rows)

            variants[name] = {
                "uses_vector_index": uses_vector_index,
                "explain": plan,
                "rows_returned": rows_returned,
                "mean_ms": mean(durations_ms),
                "stdev_ms": stdev(durations_ms),
                "p50_ms": median(durations_ms),
                "p95_ms": quantiles(
                    durations_ms,
                    n=100,
                    method="inclusive",
                )[94],
                "durations_ms": durations_ms,
            }

    cosine_mean = float(variants["cosine"]["mean_ms"])

    ratios = {
        name: float(result["mean_ms"]) / cosine_mean
        for name, result in variants.items()
        if name != "cosine" and cosine_mean > 0
    }

    return {
        "experiment": "metric_mismatch",
        "sample_query": sample_query,
        "mariadb_version": server_version,
        "chunk_count": chunk_count,
        "top_k": top_k,
        "warmup_runs": 1,
        "repeats": repeats,
        "variants": variants,
        "latency_ratios_vs_cosine": ratios,
        "notes": (
            "Exploratory measurements; do not interpret small-corpus "
            "latencies as representative Wikipedia performance."
        ),
    }
