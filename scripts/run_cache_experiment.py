"""Run Experiment E across isolated MariaDB HNSW cache settings."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, quantiles, stdev

from tests.fakes import FakeEmbedder
from wikirag.config import load_settings
from wikirag.db.repository import WikiRepository
from wikirag.experiments.cache_size import run_cache_size
from wikirag.experiments.reporting import write_experiment_report
from wikirag.models import EvalQuery

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
CACHE_SIZES = (1 * MIB, 16 * MIB, 64 * MIB)


def restart_cache_db(cache_bytes: int) -> None:
    """Recreate only the experiment container with a cache limit.

    Args:
        cache_bytes: HNSW cache limit in bytes.

    Returns:
        None: The experiment container becomes healthy.

    Raises:
        ValueError: If the cache size is not approved.
        subprocess.CalledProcessError: If Docker Compose fails.
    """
    if cache_bytes not in CACHE_SIZES:
        raise ValueError("Unsupported experiment cache size")

    environment = os.environ.copy()
    environment["HNSW_CACHE_BYTES"] = str(cache_bytes)

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.cache.yml",
            "up",
            "-d",
            "--force-recreate",
            "--wait",
            "cache-db",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def summarize_runs(
    runs: list[list[dict[str, object]]],
) -> dict[str, object]:
    """Summarize repeated measurements for one cache setting.

    Args:
        runs: Measurement rows collected after separate restarts.

    Returns:
        dict[str, object]: Timing statistics and full raw observations.

    Raises:
        ValueError: If no valid measurements are supplied.
    """
    if not runs or any(not run for run in runs):
        raise ValueError("At least one complete run is required")

    first = [float(run[0]["first_observed_ms"]) for run in runs]

    warm = [float(value) for run in runs for row in run for value in row["warm_durations_ms"]]

    return {
        "restart_cycles": len(runs),
        "first_query_after_restart_ms": first,
        "first_query_mean_ms": mean(first),
        "warm_sample_count": len(warm),
        "warm_mean_ms": mean(warm),
        "warm_stdev_ms": stdev(warm) if len(warm) > 1 else 0.0,
        "warm_p50_ms": median(warm),
        "warm_p95_ms": quantiles(
            warm,
            n=100,
            method="inclusive",
        )[94],
        "uses_vector_index": all(bool(row["uses_vector_index"]) for run in runs for row in run),
        "runs": runs,
    }


def main() -> None:
    """Benchmark three HNSW cache limits and restore the default.

    Returns:
        None: Writes reproducible JSON results after successful runs.

    Raises:
        ValueError: If command-line settings are invalid.
        subprocess.CalledProcessError: If Docker orchestration fails.
        ValidationError: If the repository rejects the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()

    if arguments.cycles < 1 or arguments.repeats < 5:
        raise ValueError("Use at least one cycle and five warm repeats")

    settings = load_settings()
    config = replace(
        settings.db,
        host="127.0.0.1",
        port=13308,
        database="wikirag_cache_test",
    )

    queries = [
        EvalQuery(
            query_id=f"cache-{index}",
            text=f"Synthetic cache benchmark question {index}",
            relevant_article_ids=frozenset({8800001}),
        )
        for index in range(5)
    ]

    measurements: dict[int, list[list[dict[str, object]]]] = {size: [] for size in CACHE_SIZES}

    try:
        for cycle in range(arguments.cycles):
            for cache_bytes in CACHE_SIZES:
                print(
                    f"\nCycle {cycle + 1}/{arguments.cycles}, cache {cache_bytes // MIB} MiB",
                    flush=True,
                )

                # Recreate only cache-db; its persistent data volume survives.
                restart_cache_db(cache_bytes)

                with WikiRepository(config) as repo:
                    if repo.count_chunks() != 6000:
                        raise ValueError("Expected exactly 6000 synthetic chunks")

                    results = run_cache_size(
                        queries,
                        repo,
                        FakeEmbedder(),
                        cache_sizes_bytes=(cache_bytes,),
                        repeats=arguments.repeats,
                        config=config,
                    )

                measurements[cache_bytes].append(results)

                avg = mean(float(row["warm_mean_ms"]) for row in results)
                print(f"Warm mean: {avg:.3f} ms", flush=True)

    finally:
        # Restore the isolated service to its normal 16 MiB setting.
        print("\nRestoring 16 MiB cache...", flush=True)
        restart_cache_db(16 * MIB)

    summaries = {
        str(cache_bytes): summarize_runs(measurements[cache_bytes]) for cache_bytes in CACHE_SIZES
    }

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    report = {
        "experiment": "cache_size",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "database": "wikirag_cache_test",
        "dataset": "deterministic synthetic random vectors",
        "dataset_seed": 20261026,
        "chunk_count": 6000,
        "embedding_dimensions": 384,
        "query_embedder": "FakeEmbedder",
        "query_ids": [query.query_id for query in queries],
        "queries_per_cycle": len(queries),
        "warm_repeats_per_query": arguments.repeats,
        "restart_cycles": arguments.cycles,
        "hardware": {
            "system": platform.system(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "cache_sizes_bytes": list(CACHE_SIZES),
        "notes": (
            "Each cycle recreates the isolated MariaDB service but "
            "preserves its database volume. First-observed queries "
            "are measured after server restart, not after clearing "
            "the operating-system file cache. Synthetic random "
            "embeddings do not measure semantic retrieval quality."
        ),
        "settings": summaries,
    }

    filename = (
        "cache_size_synthetic_smoke.json" if arguments.cycles < 5 else "cache_size_synthetic.json"
    )
    output = write_experiment_report(
        report,
        output_path=ROOT / "results" / "experiments" / filename,
    )

    print("\nExperiment E summary")
    for size in CACHE_SIZES:
        summary = summaries[str(size)]
        print(
            f"{size // MIB:>2} MiB: "
            f"mean={summary['warm_mean_ms']:.3f} ms, "
            f"p95={summary['warm_p95_ms']:.3f} ms"
        )

    print("Report saved:", output)


if __name__ == "__main__":
    main()
