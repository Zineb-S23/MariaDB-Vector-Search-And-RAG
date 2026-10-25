# MariaDB Database Design and Verification

> **Scope:** MariaDB schema, database repository, transactional ingestion, and verified test infrastructure for WikiRAG (Area 4: “Vector search and RAG, natively”). This document records implemented behavior separately from planned native vector search and experiments.
>
> **Sources of truth:** the official assignment in `Database project information.docx`; the checked-in implementations, `docs/interfaces.md`, and `src/wikirag/models.py`; the foundation handoff; and the MariaDB Area 4 architecture blueprint v0.4. The official brief takes precedence on requirements, and actual code takes precedence when reporting implementation status.

## 1. Why MariaDB?

The official Area 4 task is not merely to store embeddings. It asks us to query a real Wikipedia corpus using MariaDB 11.8's native `VECTOR` column, HNSW index and `VEC_DISTANCE_COSINE`, while applying relational conditions (such as category, article length, revision date and links) **in the same SQL statement** as semantic ranking. MariaDB also holds article metadata and embeddings in one transactional InnoDB database, avoiding synchronization with an external vector store.

**Current status:** Real database storage, retrieval of records, atomic ingestion, relationships and deletion have integration-test coverage. Semantic ranking with `VEC_DISTANCE_COSINE`, hybrid SQL filtering and recall/latency benchmarks are **planned for `feature/database-search` and are not claimed as completed here**. The blueprint v0.4 branch plan separates these deliverables; an older M1 description groups them together.

## 2. Environment and reproducibility

| Setting | Recorded configuration |
| --- | --- |
| MariaDB | Docker image `mariadb:11.8` |
| Developer's observed server build | `11.8.9-MariaDB-ubu2404` (recheck on each machine) |
| Python/package tooling | Python 3.14.4 tested locally; `uv` and the committed `uv.lock` |
| Main database | `wikirag` by default; configurable using `DB_NAME` |
| Isolated integration database | `wikirag_test` by default; configurable with `DB_NAME_TEST`; must end in `_test` |
| Docker host port | `127.0.0.1:13307` by default (container port 3306) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Storage engine | InnoDB |
| Vector index | HNSW; `VECTOR INDEX (embedding) M=16 DISTANCE=cosine` |

Never commit `.env` or passwords. Each contributor copies `.env.example` to a private `.env`; `.venv/` is the Python environment and is **not** where credentials belong. Docker Compose's named volume retains MariaDB data across container restarts. Do not use `docker compose down -v` unless intentional data deletion is acceptable.

From the repository root:

```bash
uv sync --extra dev --locked
test -f .env || cp .env.example .env  # Do not overwrite existing local secrets
make up
make version         # Prompts securely for the local MariaDB root password
make setup-test-db   # Idempotently creates/grants the separate test DB
make test-db         # Integration tests against DB_NAME_TEST
make test-unit
make lint
```

Do not overwrite an existing `.env` with the `cp` command. `scripts/create-test-db.sh` checks the test database name, runs Docker Compose, grants the application user access to the isolated database and verifies a connection. It must **never** drop or reset the main database.

To initialize or verify the **main database** schema, from the project root, use the existing schema API rather than reimplementing table creation:

```bash
uv run --extra dev python - <<'PY'
from contextlib import closing

from wikirag.config import load_settings
from wikirag.db.connection import get_connection
from wikirag.db.schema import init_schema, verify_schema

with closing(get_connection(load_settings().db)) as conn:
    init_schema(conn)
    verify_schema(conn)

print("Main database schema verified")
PY
```

This calls `init_schema` in normal (`recreate=False`) mode. Never pass `recreate=True` to the main database. Use `sql/001_schema.sql` as the canonical DDL and `verify_schema` to detect schema drift.

## 3. Schema and relationships

```text
embedding_meta                     articles
(model_name, embedding_dim)         (article_id PK; title UNIQUE; metadata)
                                            |
                         +------------------+-------------------+
                         |                  |                   |
                       chunks       article_categories     links (outgoing)
                (article_id FK)       (article_id FK)      (source_article_id FK)
                         |                  |                   |
                    VECTOR(384)        categories         target_article_id
                    HNSW cosine     (category_id PK)       (nullable FK to articles)
```

There are **six tables**, all using InnoDB:

### `embedding_meta`

A single metadata row (`id=1`, enforced by a check constraint) records `model_name`, `embedding_dim` and `created_at`. This guards against mixing embeddings from different models or dimensions and makes vector-search results interpretable. Repository method `ensure_embedding_meta(model_name, dim)` is idempotent for matching metadata and rejects incompatible metadata.

### `articles`

- `article_id`: Wikipedia's stable page ID; natural primary key shared with snapshots and evaluation labels.
- `title`: source article title; unique with `utf8mb4_bin` collation to preserve case-sensitive uniqueness.
- `url`: original Wikipedia page URL.
- `revision_id` and `revised_at`: source revision identity/time; `revised_at` is used for edited-since filters. Python records use naive UTC timestamps by the shared contract.
- `length_chars`: cleaned article length, used by length filters.
- `fingerprint`: SHA-256 hex fingerprint of ingestion-relevant content/settings for skipping unchanged articles.
- `topic`: optional coarse corpus label, useful for SQL filtering and experiments.
- `language`: two-character language label; defaults to `en`.
- `fetched_at` and `ingested_at`: acquisition and database write timestamps for provenance.

Indexes: primary key on ID; unique title index; secondary indexes for `topic`, `revised_at`, and `length_chars`. These support typical metadata predicates. The fingerprint must be computed consistently by the ingestion pipeline, including relevant text/chunking/model choices.

### `chunks`

- `chunk_id`: database-assigned primary key, used for search hits and citations.
- `article_id`: foreign key to `articles`; `ON DELETE CASCADE` prevents orphaned chunks.
- `chunk_index`: zero-based position within its article; unique with `article_id` to prevent duplicate positions and enable neighbor lookup.
- `section_title`: optional heading for contextual display.
- `text`: stored chunk text, and `char_count`: the text length.
- `embedding`: `VECTOR(384) NOT NULL`, with one `VECTOR INDEX (embedding) M=16 DISTANCE=cosine`.

The model's dimension determines the `VECTOR(384)` declaration. Cosine distance is specified explicitly rather than relying on an index default. `M=16` is a **starting index configuration**, not a demonstrated optimum: the planned index-versus-exact experiment must measure its recall/latency trade-off. The vector type and application validation jointly guard against invalid dimensions.

### `categories`

A normalized dictionary of category names (`category_id` primary key, case-sensitive unique `name`). Storing categories in a separate table enables relational predicates and avoids maintaining an opaque JSON list per article.

### `article_categories`

Many-to-many bridge between articles and categories. Composite primary key `(article_id, category_id)` prevents duplicate memberships. A category-side index supports joins/filtering. Cascading foreign keys remove memberships when their article or category is deleted; category dictionary rows themselves may remain after an article update.

### `links`

Stores outgoing Wikipedia links as `(source_article_id, target_title, target_article_id)`:

- `(source_article_id, target_title)` is the primary key, preserving outgoing link titles even if targets are outside the ingested corpus.
- `source_article_id` references `articles` with `ON DELETE CASCADE`.
- `target_article_id` is nullable, references `articles` and uses `ON DELETE SET NULL`; target IDs are populated by `resolve_links()` when the corresponding article exists locally.
- An index on `target_article_id` supports navigation to linked articles.

This design supports the official assignment's “linked from another article” predicate through relational joins.

## 4. Repository API and transaction semantics

The canonical public signatures are in [`docs/interfaces.md`](interfaces.md). Ingestion and retrieval modules must use `WikiRepository` rather than importing PyMySQL or issuing their own SQL. The DB layer does **not** import the ingestion, embedding, retrieval or RAG modules. Shared boundary values use dataclasses from `src/wikirag/models.py`, including `ArticleRecord`, `ChunkRecord`, `StoredChunk`, `SearchFilters`, and `SearchHit`.

Implemented and exercised against MariaDB locally:

- Connection lifecycle and `with WikiRepository(...)` support; connectivity and schema checks.
- `ensure_embedding_meta`, `get_fingerprint`, `count_articles`, and `count_chunks`.
- `ingest_article`: validate inputs, begin transaction, compare fingerprint with a locking read, return `"skipped"` without changing identical content, or atomically insert/update the article and replace chunks, category memberships and outgoing links; return `"inserted"` or `"updated"`.
- `get_article`, `get_article_by_title`, `get_chunk`, `get_neighbor_chunks`, `list_topics`, `list_categories`.
- `resolve_links`, `get_linked_articles`, `delete_article`.

A failed SQL operation rolls back the article's metadata and related rows together. Tests include an intentional failure late in a write sequence and verify that the preexisting data remains intact. Article deletion cascades to its chunks/outgoing link and category-association rows; incoming links keep the title but have their resolved target ID set to `NULL`.

**Follow-up compatibility check:** `get_article_by_title()` and `resolve_links()` currently use SQL `LOWER(...)` comparisons, while the in-memory fake uses Python `casefold()`. Those operations are not identical for every Unicode string, and binary title uniqueness also permits titles distinguished only by case. Ambiguous/case-sensitive Wikipedia titles require deliberate policy and additional tests before claiming complete fake-versus-real equivalence.

## 5. Verified evidence (not benchmark results)

The following results were reported from the developer's local WSL/Docker environment:

- MariaDB 11.8 Docker container reached `healthy` status, and the SQL schema was applied; `SHOW CREATE TABLE chunks` was inspected for the native `VECTOR(384)` and cosine index.
- Real insertion tests passed using `VEC_FromText` and verified native vector round-tripping with `VEC_ToText`.
- Real integration tests passed for model metadata, insert/skip/update, category/link persistence, read operations, link resolution/deletion, invalid vector rejection and transactional rollback.
- `make setup-test-db` succeeded and printed the separate `wikirag_test` database.
- Latest reported local suite: **165 passed** (141 unit tests and 24 MariaDB integration tests); Ruff lint and format checks passed, with `uv lock --check` succeeding.

These are **local** observations. They are not a claim that CI has run on the GitHub branch, that a real Wikipedia corpus has been ingested, or that native vector ranking and hybrid-query performance have been demonstrated. The separate GitHub Actions MariaDB job must be verified on a pushed PR.

For reproducible database evidence, capture the real output of:

```bash
make version
make setup-test-db
make test-db
make lint

# Log in interactively; enter your own local password when prompted.
docker compose exec db mariadb -u wikirag -p wikirag -e 'SHOW CREATE TABLE chunks\G'
docker compose exec db mariadb -u wikirag -p wikirag -e "SHOW VARIABLES LIKE 'mhnsw%';"
```

Do not paste private passwords or replace a missing output with an expected-looking value. The exact MariaDB patch release and live `mhnsw%` variable values must come from the running server.

## 6. MariaDB native-vector spike and planned follow-up

The blueprint calls for recording **measured** answers to four spike questions. The current evidence supports creation and use of the vector column/index and successful storage; it does **not yet establish** the following findings:

| Spike question | Present evidence | Next verification |
| --- | --- | --- |
| Do SQL topic/category filters and joins affect HNSW top-K completeness? | Not measured | Run `EXPLAIN` and compare counts/rankings of Q1–Q3 under selective filters on populated Wikipedia chunks. |
| How do we reliably force an exact cosine scan for evaluation? | Not measured | Test no-index or index-defeating query shapes and verify plans with `EXPLAIN`. |
| What `mhnsw%` variable values does the actual MariaDB server use? | Command prepared; values not captured in this document | Record `SHOW VARIABLES LIKE 'mhnsw%'` output, server version and date. |
| Can FULLTEXT coexist with the vector index? | Not tested | Optional keyword-baseline experiment; do not block mandatory vector/hybrid search on it. |

The upcoming `feature/database-search` deliverable must implement `search_chunks`, `search_chunks_exact`, and `set_ef_search`, test all eight `SearchFilters` fields and SQL joins, and provide `sql/002_example_queries.sql` (Q1–Q6). In particular, HNSW top-K with selective SQL predicates can return too few eligible rows; the public repository contract requires `min(top_k, eligible_matching_chunks)` matches unless `max_distance` removes them. The implementation and measurements must prove how this is handled (for example, over-fetching or falling back to exact ranking).

Only after running the relevant experiments should this section contain actual `EXPLAIN` plans, recall, latency, selectivity counts, or claims that one configuration outperforms another. The official evaluation requires grounded Wikipedia queries, recall and latency—not a qualitative assertion that search is “good.”

## 7. Verification and maintenance checklist

- The schema in `sql/001_schema.sql`, `verify_schema()`, repository SQL and this document must agree.
- Keep `EMBEDDING_DIM=384` and the model name synchronized across ingestion/query embedding generation and `embedding_meta`.
- Keep database integration tests isolated to `DB_NAME_TEST`; never recreate the main database during tests.
- Run `make test-unit`, `make test-db`, `make lint`, and `uv lock --check` before a database PR; inspect GitHub Actions results after pushing.
- Preserve signatures and return types in `docs/interfaces.md`, and add fake-versus-real repository contract tests before calling the repository implementation fully interchangeable.
- Record actual MariaDB version, HNSW parameters and experiment methodology with any published performance claims.
- Update README setup/reproduction instructions together with production features, and merge reviewed PRs into `main` because the official evaluator reads `main`, not unmerged branches.

