# MariaDB Native-Vector Depth Experiments — C, D, and E

> **Project:** WikiRAG, Area 4 — *Vector search and RAG, natively*.  
> **Evidence date:** 26 October 2026.  
> **Scope:** MariaDB-native index planning, transactional visibility, and HNSW cache-size behavior. These tests complement—but do not replace—the official assignment's real-Wikipedia retrieval and evaluation requirements.

## Environment, contracts, and controls

| Property | Verified or configured value |
| --- | --- |
| MariaDB server | `11.8.9-MariaDB-ubu2404`, Docker `mariadb:11.8` |
| Language / development environment | Python 3.14.4 with `uv` on WSL/Linux |
| Storage | MariaDB/InnoDB, `chunks.embedding VECTOR(384)` |
| Index | Native HNSW, index name `embedding`, `M=16`, `DISTANCE=cosine` |
| Similarity function | `VEC_DISTANCE_COSINE` |
| Search breadth | `mhnsw_ef_search=20` for Experiment E |
| Normal development database | `wikirag`, host port 13307 |
| Destructive integration-test database | `wikirag_test`; separate from application data |
| Cache benchmark database | `wikirag_cache_test`, isolated `cache-db` service on host port 13308, separate Docker volume |

`WikiRepository.search_chunks()` uses the MariaDB HNSW-eligible rank-first, join-second SQL query. Its filtered mode deliberately performs exact cosine ranking after relational predicates, using `IGNORE INDEX (embedding)` to prevent selective-filter top-K under-fill. `WikiRepository.search_chunks_exact()` supplies an exact comparison baseline. Shared API signatures and return types remain defined in [`interfaces.md`](interfaces.md).

The evaluation-specific synthetic datasets and `FakeEmbedder` are **not** a substitute for Zineb's real Wikipedia corpus and Marwan's labeled retrieval evaluation. A synthetic query can exercise the HNSW path without having semantic relevance to the inserted vector text.

## Experiment C — Distance metric mismatch

**Question.** Does a query use the cosine HNSW index when its distance expression changes?

**Code:** `src/wikirag/experiments/metric_mismatch.py` (`run_metric_mismatch`), `src/wikirag/db/diagnostics.py` (`explain_search`), and SQL plan checks in `tests/integration/test_repository.py`.

**Method.** For an identical vector and top-K limit, `EXPLAIN` and time these three forms: (1) `ORDER BY VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) ASC`, (2) `ORDER BY VEC_DISTANCE_EUCLIDEAN(...) ASC`, and (3) `ORDER BY (1 - VEC_DISTANCE_COSINE(...)) DESC`. The experiment warms up each variant and measures at least five repetitions, retaining plans, individual durations, mean, standard deviation, p50/p95, and latency ratios relative to cosine.

**Observed on the small synthetic verification dataset:** the direct cosine ranking selected the `embedding` VECTOR index, while the Euclidean and wrapped cosine expressions did not. This shows why the exact SQL distance/order expression and the execution plan must be checked rather than merely assuming that the vector column causes HNSW acceleration. These are plan-selection observations, **not** general latency benchmarks or real-Wikipedia recall measurements. Do not publish numerical C latency ratios without a committed raw experiment report for that run.

**Interpretation.** The production semantic path preserves the direct cosine ranking expression in an inner query before joining article metadata. The exact baseline intentionally avoids the index. Optimizer plans must be rechecked on the real Wikipedia corpus.

## Experiment D — Transactional visibility and deletion regression

**Question.** Do independently connected sessions observe article metadata and vector-search results consistently before commit, after rollback, and after commit?

**Code:** `src/wikirag/db/diagnostics.py` (`demo_transaction_visibility`), `src/wikirag/experiments/transaction_visibility.py` (`run_transaction_visibility`), `src/wikirag/db/repository.py` (`ingest_article`, `delete_article`), and `tests/integration/test_repository.py`.

**Method.** Two independent MariaDB connections are configured for each of `REPEATABLE READ` and `READ COMMITTED`. The writer inserts an article and indexed vector without committing; the reader checks four independently recorded observations: article count, chunk count, indexed Q1 hits, and exact-cosine hits. After rollback, the insertion is repeated and committed. A newly opened third connection also checks committed visibility. The reader's transaction is ended after each observation to avoid comparing stale snapshots across phases. Synthetic rows are cleaned up after each isolation-level run.

| Observation | Articles | Chunks | HNSW hits | Exact hits |
| --- | ---: | ---: | ---: | ---: |
| Before insertion | 0 | 0 | 0 | 0 |
| Writer uncommitted | 0 | 0 | 0 | 0 |
| After writer rollback | 0 | 0 | 0 | 0 |
| Before writer commit on second insertion | 0 | 0 | 0 | 0 |
| After writer commit | 1 | 1 | 1 | 1 |
| After commit, fresh reader | 1 | 1 | 1 | 1 |
| After cleanup | 0 | 0 | 0 | 0 |

Both isolation levels produced this pattern in the final combined test. The combined experiment was repeated **five times**, with `indexed_search_misses=[]` on every run, and the regression test now uses strict assertions rather than a permanent `xfail`.

### Failure discovered and mitigation

During earlier runs, deletion of a parent `articles` row via `ON DELETE CASCADE` sometimes preceded a later state where the next committed vector was visible to relational counts and exact cosine search but absent from HNSW top-K results. The effect followed the *second experiment on a reused index*, not a particular isolation level. A fresh index for each isolation level had passed independently.

**Tested mitigation:** explicitly delete the child's indexed `chunks` rows **before** deleting its parent `articles` row, in the same database transaction. This ordering is now used by `WikiRepository.delete_article()`; `ingest_article()` likewise explicitly removes stale chunks when replacing an article's revision. The foreign-key `ON DELETE CASCADE` remains in the schema as an integrity safety net. The mitigation also passed 10 repeated article insert/search/delete cycles and 10 repeated article-update/search cycles through the actual repository.

**Limit:** this is a reproducible local observation and a verified mitigation on MariaDB 11.8.9, **not** proof of the upstream MariaDB root cause or a guarantee that all HNSW mutation histories are unaffected. Retain both indexed and exact-search regression assertions, and retest after MariaDB version changes.

Re-run relevant integration tests (this command uses the destructive *dedicated* `wikirag_test` fixture; never target the application database):

```bash
uv run --extra dev python -m pytest tests/integration/test_repository.py \
  -k "transaction_visibility or repeated_article_deletion or repeated_article_updates" -v
```

## Experiment E — HNSW cache-size latency

**Question.** How does MariaDB's `mhnsw_max_cache_size` affect native indexed-search latency?

**Implementation:** `docker-compose.cache.yml`, `scripts/seed_cache_experiment.py`, `scripts/run_cache_experiment.py`, `src/wikirag/experiments/cache_size.py`, `tests/unit/test_cache_experiment.py`, and the committed machine-readable report [`../results/experiments/cache_size_synthetic.json`](../results/experiments/cache_size_synthetic.json).

### Corpus and measurement protocol

- **Corpus:** one synthetic article with 6,000 deterministically generated, normalized, 384-dimensional random vectors; seed `20261026`.
- **Query set:** five fixed `FakeEmbedder` questions; query embedding time is excluded from timed database calls.
- **Configurations:** `1 MiB = 1,048,576`, `16 MiB = 16,777,216`, and `64 MiB = 67,108,864` bytes.
- **Index and SQL:** `WikiRepository.search_chunks(..., top_k=5)`, MariaDB native cosine HNSW; `EXPLAIN` verified `key=embedding` for every measured query.
- **Controlled search setting:** session `mhnsw_ef_search=20`.
- **Replication:** five cycles per configuration; before every cycle/configuration the **isolated** `cache-db` container is recreated and its persistent data volume retained. Each cycle runs five fixed queries with five timed warm repetitions each: **125 warm samples per cache setting**, 375 warm samples overall. One first-observed query after each server restart is tracked separately for each configuration.
- **Statistics:** report raw per-query durations, mean, sample standard deviation, median (p50), inclusive p95, first-observed latency, Docker restart cycle, MariaDB version, corpus size and settings. The benchmark restores the container's default 16 MiB limit in a `finally` block.
- **Reproducibility metadata:** the report records its implementation Git commit `015028c93d067df48358a0ec98a7f1eaff4d4db2`, generation time `2026-10-26T19:06:47+00:00`, synthetic seed, host characteristics, index plans and full samples. The report commit itself follows the implementation commit.

### Verified final results, 26 October 2026

| `mhnsw_max_cache_size` | Mean warm query latency | Warm p95 | Warm samples | HNSW used |
| --- | ---: | ---: | ---: | --- |
| 1 MiB | **6.018 ms** | 8.516 ms | 125 | Yes |
| 16 MiB | **2.878 ms** | 5.023 ms | 125 | Yes |
| 64 MiB | **3.016 ms** | 4.683 ms | 125 | Yes |

In this run, 16 MiB had approximately **52% lower mean latency** than 1 MiB; 64 MiB had approximately **50% lower mean latency** than 1 MiB. The 16 MiB configuration was slightly faster on mean, whereas 64 MiB had the lowest p95. Across individual cycles, neither 16 nor 64 MiB was uniformly best, so do **not** declare a universal optimum. Read precise values and raw variance from the committed JSON; the table is rounded to three decimals.

### Reproduction

The instructions below require a local private `.env` with valid MariaDB credentials and working Docker/`uv`. They use **port 13308 only** for benchmark data. No benchmark step requires changing the main development database's global cache setting.

```bash
uv sync --extra dev --locked

docker compose -f docker-compose.yml -f docker-compose.cache.yml \
  up -d --wait cache-db
```

On a new, empty `wikirag_cache_test` database, initialize the canonical schema non-destructively:

```bash
uv run --extra dev python - <<'PY'
from contextlib import closing
from dataclasses import replace
from wikirag.config import load_settings
from wikirag.db.connection import get_connection
from wikirag.db.schema import init_schema

config = replace(load_settings().db, host="127.0.0.1", port=13308,
                 database="wikirag_cache_test")
with closing(get_connection(config)) as connection:
    init_schema(connection, recreate=False)
PY
```

Seed **only if empty** (the script refuses to overwrite a non-empty corpus):

```bash
uv run --extra dev python scripts/seed_cache_experiment.py
```

Run the full five-cycle comparison:

```bash
uv run --extra dev python -m scripts.run_cache_experiment --cycles 5 --repeats 5
```

Results are regenerated to `results/experiments/cache_size_synthetic.json`. **Rerunning the benchmark changes the timings and overwrites this file**, so compare differences before committing new numbers. The benchmark's host must have enough memory for the 64 MiB configuration.

### Interpretive limitations

1. **First query after MariaDB restart ≠ fully cold cache.** The MariaDB process is recreated, but the operating-system file cache is not explicitly cleared. The report calls this `first_query_after_restart_ms`, not a proven cold-storage latency.
2. **Synthetic data and synthetic queries.** These timings verify an indexed HNSW path and cache setting effects under synthetic load, not Wikipedia question-answering accuracy or semantic similarity quality.
3. **Limited scale and environment.** One small corpus and one local Docker/WSL host do not characterize throughput, concurrent clients, alternative hardware, or all HNSW index sizes.
4. **Potential run order effects.** The program cycles through 1 → 16 → 64 MiB; it does not randomize settings, so residual host activity and order-dependent effects may contribute to variation. The saved per-cycle timings allow review.
5. **Performance versus recall.** All queries returned rows and used the index, but this experiment does not measure nearest-neighbor recall against the exact baseline.

## Verification status and remaining work

The last reported local checks following the implementation passed **191 unit tests and 105 MariaDB integration tests (296 total)**, plus Ruff lint/format and `git diff --check`. This is *local evidence*; it does not establish passing GitHub Actions or tests on merged `main`.

**Still required for the official assignment and the architecture blueprint:** ingest a real reproducible Wikipedia corpus; run a frozen labeled question set and publish exact-versus-HNSW recall and latency; demonstrate Q1–Q6 against populated data, including relational predicates and joins; complete the remaining filter-selectivity and `ef_search` experiments where assigned; publish the findings and commands in the root README; obtain teammate review; merge and verify on `main`. The official Word assignment takes precedence over optional blueprint depth experiments.
