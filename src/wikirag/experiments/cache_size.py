"""Experiment E: measure HNSW search under one cache setting."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from statistics import mean, median, quantiles, stdev
from time import perf_counter
from typing import TYPE_CHECKING

from wikirag.config import DbConfig
from wikirag.db.connection import get_connection
from wikirag.db.queries import SEARCH_CHUNKS_SEMANTIC
from wikirag.db.repository import WikiRepository
from wikirag.errors import ValidationError
from wikirag.models import EvalQuery

if TYPE_CHECKING:
    from wikirag.embeddings import Embedder


CACHE_DATABASE = "wikirag_cache_test"
CACHE_PORT = 13308
MIB = 1024 * 1024


def run_cache_size(
    queries: Sequence[EvalQuery],
    repo: WikiRepository,
    embedder: Embedder,
    *,
    cache_sizes_bytes: Sequence[int],
    repeats: int = 5,
    config: DbConfig | None = None,
) -> list[dict[str, object]]:
    """Measure HNSW search latency at the active cache limit.

    Run separately for each cache configuration. A later
    orchestration script will restart the isolated server between
    settings and combine the measurements.

    Args:
        queries: Fixed evaluation queries used in every comparison.
        repo: Repository connected to the isolated cache database.
        embedder: Shared query embedding implementation.
        cache_sizes_bytes: One expected active cache size in bytes.
        repeats: Number of warm measurements per query, at least five.
        config: Dedicated experiment database connection settings.

    Returns:
        list[dict[str, object]]: Per-query search latency statistics,
            index plans, database version and cache configuration.

    Raises:
        ValidationError: If inputs, database, index or cache settings
            do not satisfy the experiment requirements.
        DatabaseConnectionError: If MariaDB is unavailable.
        DatabaseError: If database operations fail.
    """
    if not queries or any(
        not isinstance(query, EvalQuery) or not query.text.strip() for query in queries
    ):
        raise ValidationError("Provide at least one valid EvalQuery")

    if type(repeats) is not int or repeats < 5:
        raise ValidationError("repeats must be at least five")

    if (
        len(cache_sizes_bytes) != 1
        or type(cache_sizes_bytes[0]) is not int
        or cache_sizes_bytes[0] < MIB
    ):
        raise ValidationError("Run one cache limit of at least 1 MiB at a time")

    selected = config if config is not None else repo._config

    # Never benchmark the normal application or integration-test database.
    if (
        selected.host != "127.0.0.1"
        or selected.port != CACHE_PORT
        or selected.database != CACHE_DATABASE
        or repo._config != selected
    ):
        raise ValidationError("Experiment E requires the isolated database on port 13308")

    expected_bytes = cache_sizes_bytes[0]

    with closing(get_connection(selected)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT @@GLOBAL.mhnsw_max_cache_size AS cache_bytes, VERSION() AS server_version"
        )
        server = cursor.fetchone()

        actual_bytes = int(server["cache_bytes"])

        if actual_bytes != expected_bytes:
            raise ValidationError(f"Expected cache limit {expected_bytes}, found {actual_bytes}")

        cursor.execute("SELECT COUNT(*) AS total FROM chunks")
        chunk_count = int(cursor.fetchone()["total"])

        if chunk_count == 0 or repo.count_chunks() != chunk_count:
            raise ValidationError("The cache experiment corpus must be populated")

        # Keep search breadth identical across cache configurations.
        repo.set_ef_search(20)

        results: list[dict[str, object]] = []

        for query in queries:
            # Embed before timing so model latency cannot skew SQL results.
            vector = embedder.embed_query(query.text)
            vector_json = WikiRepository._serialize_search_vector(vector)
            params = (vector_json, vector_json, 5)

            # Inspect the same SQL used by production semantic retrieval.
            cursor.execute(
                "EXPLAIN " + SEARCH_CHUNKS_SEMANTIC,
                params,
            )
            plan = [dict(row) for row in cursor.fetchall()]

            index_used = any(row.get("key") == "embedding" for row in plan)

            if not index_used:
                raise ValidationError(f"HNSW index not selected for {query.query_id}")

            # The first observation is not assumed to be fully cold.
            started = perf_counter()
            initial_hits = repo.search_chunks(vector, top_k=5)
            first_ms = (perf_counter() - started) * 1000.0

            if not initial_hits:
                raise ValidationError("Indexed search returned no results")

            durations_ms: list[float] = []

            # Repeated queries measure the warmed search path.
            for _ in range(repeats):
                started = perf_counter()
                hits = repo.search_chunks(vector, top_k=5)
                elapsed_ms = (perf_counter() - started) * 1000.0

                if not hits:
                    raise ValidationError("Indexed search returned no results")

                durations_ms.append(elapsed_ms)

            results.append(
                {
                    "query_id": query.query_id,
                    "cache_bytes": actual_bytes,
                    "mariadb_version": str(server["server_version"]),
                    "chunk_count": chunk_count,
                    "ef_search": 20,
                    "top_k": 5,
                    "uses_vector_index": True,
                    "explain": plan,
                    "first_observed_ms": first_ms,
                    "warm_repeats": repeats,
                    "warm_durations_ms": durations_ms,
                    "warm_mean_ms": mean(durations_ms),
                    "warm_stdev_ms": stdev(durations_ms),
                    "warm_p50_ms": median(durations_ms),
                    "warm_p95_ms": quantiles(
                        durations_ms,
                        n=100,
                        method="inclusive",
                    )[94],
                }
            )

    return results
