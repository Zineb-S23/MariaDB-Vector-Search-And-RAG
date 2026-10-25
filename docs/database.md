# MariaDB Database Design and Verification

> **Scope:** MariaDB schema, database repository, transactional ingestion, and verified test infrastructure for WikiRAG (Area 4: “Vector search and RAG, natively”). This document records implemented behavior separately from planned native vector search and experiments.
>
> **Sources of truth:** the official assignment in `Database project information.docx`; the checked-in implementations, `docs/interfaces.md`, and `src/wikirag/models.py`; the foundation handoff; and the MariaDB Area 4 architecture blueprint v0.4. The official brief takes precedence on requirements, and actual code takes precedence when reporting implementation status.

## 1. Why MariaDB?

The official Area 4 task is not merely to store embeddings. It asks us to query a real Wikipedia corpus using MariaDB 11.8's native `VECTOR` column, HNSW index and `VEC_DISTANCE_COSINE`, while applying relational conditions (such as category, article length, revision date and links) **in the same SQL statement** as semantic ranking. MariaDB also holds article metadata and embeddings in one transactional InnoDB database, avoiding synchronization with an external vector store.



**Current status:** Real database storage, atomic ingestion,
relationships, deletion, native semantic retrieval, and hybrid SQL
filtering have integration-test coverage.

Unfiltered semantic search uses an HNSW-index-eligible MariaDB query.
Hybrid searches combine relational SQL predicates with exact cosine
ranking to avoid selective-filter top-K under-fill.

The real repository also implements `search_chunks_exact()` as an
exact nearest-neighbor evaluation baseline and `set_ef_search()` for
session-scoped HNSW tuning.

All public database-search methods are now implemented. Recall,
latency, and HNSW parameter experiments on a representative Wikipedia
corpus remain outstanding.


Unfiltered searches use an HNSW-index-eligible query with
`VEC_DISTANCE_COSINE`. Filtered searches apply relational SQL
constraints and exact cosine ranking in one SQL statement, avoiding
under-filled top-K results caused by filtering only after an HNSW
search.

The fake and MariaDB repositories have passed the same hybrid-filter
parity tests. The exact-search evaluation API, HNSW parameter tuning,
and recall/latency benchmarks remain planned.



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

**Title matching policy:** get_article_by_title() and resolve_links() use exact, case-sensitive title matching, consistent with the database's binary title collation. The in-memory fake follows the same policy. The Wikipedia ingestion pipeline must normalize article titles and outgoing link titles consistently before storage.


## 5. Verified evidence (not benchmark results)

The following results were reported from the developer's local WSL/Docker environment:

- The MariaDB 11.8 Docker container reached `healthy` status, and the SQL schema was applied successfully.
- `SHOW CREATE TABLE chunks` confirmed the native `VECTOR(384)` column and cosine-distance VECTOR index.
- Real database insertion tests passed using `VEC_FromText`, and vector round-tripping was verified using `VEC_ToText`.
- Integration tests verified embedding metadata, atomic article ingestion, insertion/skipping/updating, category and link persistence, article retrieval, link resolution, deletion, invalid-vector rejection, and transaction rollback.
- `make setup-test-db` successfully configured the separate `wikirag_test` database.
- Native semantic search was tested against MariaDB using stored synthetic embeddings. Tests verified cosine-distance ranking, top-K limits, article metadata, and similarity scores.
- Hybrid SQL filtering was tested against MariaDB for topics, categories, included and excluded article IDs, article length, revision date, outgoing links, and maximum cosine distance.
- Hybrid-search tests verified individual filters, combined predicates, case-insensitive topic/category matching, and filtering before top-K selection.
- Fake-versus-MariaDB parity tests verified consistent filtered-search behavior between `InMemoryRepository` and `WikiRepository`.
- Shared `SearchFilters` validation tests verified invalid types, identifiers, thresholds, timestamps, and boundary values. Both repository implementations reject invalid filters using `ValidationError`.
- Exact cosine retrieval was verified against MariaDB, including ranking, optional filters, invalid inputs, and exclusion of the vector index.
- `EXPLAIN` integration tests confirmed that the semantic-search query selects the `embedding` VECTOR index and the exact-search query does not select it.
- Session-scoped HNSW tuning was verified for accepted and rejected values, connection isolation, and unchanged global settings.
- The latest reported local test suite passed **273 tests: 182 unit tests and 91 integration tests**.
- Ruff lint, Ruff formatting, and Git whitespace checks passed.

These are **local verification results**, not measurements of retrieval quality or performance on a real Wikipedia corpus.

The tests demonstrate functional database behavior using synthetic embeddings and controlled test articles. They do not establish recall, latency, scalability, or an optimal HNSW configuration.

The results also do not imply that the GitHub Actions workflows have passed for the final pull request. CI results must be checked separately after pushing the completed branch.

### Reproducing database verification

From the repository root:

```bash
uv lock --check
make version
make setup-test-db
make lint
make test-unit
make test-db
git diff --check
```

To inspect the actual MariaDB configuration:

```bash
# Enter the local MariaDB password interactively.
docker compose exec db mariadb -u wikirag -p wikirag -e 'SHOW CREATE TABLE chunks\G'

docker compose exec db mariadb -u wikirag -p wikirag -e "SHOW VARIABLES LIKE 'mhnsw%';"
```

The database version, index definition, and session variables should be recorded from the running database rather than assumed from configuration files.

Do not commit `.env`, passwords, or other private credentials.

## 6. MariaDB native-vector verification and planned experiments

### 6.1 Verified MariaDB configuration

The following configuration was inspected directly on 25 October 2026 using the running Docker MariaDB instance.

| Setting | Verified value |
| --- | --- |
| MariaDB version | `11.8.9-MariaDB-ubu2404` |
| Vector column | `VECTOR(384)` |
| Index type | Native MariaDB VECTOR/HNSW |
| Index distance | `cosine` |
| Index M parameter | `16` |
| `mhnsw_default_distance` | `euclidean` |
| `mhnsw_default_m` | `6` |
| `mhnsw_ef_search` | `20` |
| `mhnsw_max_cache_size` | `16777216` bytes (16 MiB) |

The configuration was verified using:

```sql
SELECT VERSION();
SHOW VARIABLES LIKE 'mhnsw%';
SHOW CREATE TABLE chunks\G
```

The `chunks` table contains:

```sql
embedding VECTOR(384) NOT NULL
```

MariaDB reports its vector index as:

```sql
VECTOR KEY `embedding` (`embedding`) `M`='16' `DISTANCE`='cosine'
```

Our schema explicitly specifies `M=16` and cosine distance. Therefore, the server defaults of `M=6` and Euclidean distance do not determine the configuration of this index.

The index was successfully created and verified on MariaDB 11.8.9.

Integration tests also confirmed native vector insertion using `VEC_FromText` and round-trip retrieval using `VEC_ToText`.

These observations verify the schema, native vector storage, and index configuration. They do not establish the performance or accuracy of approximate HNSW retrieval.

### 6.2 Semantic and exact search execution-plan verification

During semantic-search development, three query structures were examined using `EXPLAIN` on MariaDB 11.8.9:

1. **Minimal vector search:** MariaDB selected the native `embedding` VECTOR index (`type=index`, `key=embedding`).
2. **Vector search with an article join:** The original query used a full scan of `chunks` (`type=ALL`, `key=NULL`) and filesort.
3. **Rank-first, join-second search:** A derived-table query ranked chunks before joining article metadata. The inner query selected the native `embedding` VECTOR index.

The third structure was selected for `WikiRepository.search_chunks()` because it allows native vector-index selection while returning article titles, URLs, and chunk metadata.

**Initial limitation:** The database contained zero chunks during these first `EXPLAIN` comparisons. These initial observations established optimizer plan selection but did not demonstrate behavior with stored vectors.

**Additional execution-plan verification**

A subsequent MariaDB integration test inserted synthetic chunks into the test database and examined the execution plan for `SEARCH_CHUNKS_SEMANTIC`.

The `EXPLAIN` result confirmed that MariaDB selected the `embedding` VECTOR index for the inner nearest-neighbor ranking query.

A separate integration test examined the exact-search SQL generated by `build_hybrid_search_query()`.

That query explicitly includes:

```sql
FROM chunks AS c IGNORE INDEX (embedding)
```

The corresponding `EXPLAIN` test confirmed that the exact-search query did not select the `embedding` index.

Together, these tests establish two distinct query paths:

- **Indexed semantic search:** The tested execution plan selects MariaDB's native vector index.
- **Exact cosine search:** The tested execution plan avoids the vector index and calculates exact cosine-distance rankings.

Both query paths use MariaDB's native `VEC_DISTANCE_COSINE` function.

**Remaining limitation:** Execution-plan verification does not establish actual search latency, HNSW recall, or performance at scale.

The plans and performance must be reevaluated using a representative Wikipedia corpus.

### 6.3 Experiments still required

| Experiment | Current status | Planned verification |
| --- | --- | --- |
| Indexed hybrid SQL filtering | Not implemented or benchmarked | Investigate HNSW-compatible filtered-query plans and compare with the existing exact-filtered implementation |
| HNSW top-K under selective filters | Not measured on a representative corpus | Compare eligible result counts under increasing filter selectivity |
| Exact cosine search baseline | Implemented and integration-tested | Compare exact results with HNSW on the same Wikipedia corpus |
| HNSW vs. exact recall and latency | Not measured | Run repeated searches over a fixed labeled question set |
| HNSW `ef_search` sensitivity | Session tuning implemented; retrieval effects not measured | Compare recall and latency across multiple `ef_search` settings |
| Vector index parameter comparison | Starting configuration verified; alternatives not measured | Compare selected supported HNSW configurations under controlled conditions |
| Transaction consistency | Ingestion rollback tested; concurrent search visibility not measured | Demonstrate consistent visibility of vector and relational metadata during transactions |
| FULLTEXT compatibility | Not tested; optional | Investigate only after mandatory native-vector and hybrid-search requirements are satisfied |

These experiments belong to the database-search evaluation and final integration stages.

The official Area 4 assignment requires:

- A real Wikipedia corpus.
- MariaDB-native semantic retrieval using `VEC_DISTANCE_COSINE`.
- Relational SQL constraints combined with semantic retrieval in the same statement.
- A labeled question set with known relevant source articles.
- Measured retrieval recall and latency.

The current database implementation supports the required retrieval operations, but the Wikipedia-based evaluation results have not yet been produced.

No recall, latency, or performance conclusions are claimed at this stage.

### 6.4 Hybrid SQL filtering implementation

Milestone 2 implements database-native hybrid retrieval through `WikiRepository.search_chunks()`.

The implementation uses MariaDB's native `VEC_DISTANCE_COSINE` function alongside ordinary SQL predicates and relational joins.

Two query paths are supported:

1. **Unfiltered semantic search:** Uses the rank-first, join-second SQL query designed to allow HNSW index selection before retrieving article metadata.
2. **Filtered hybrid search:** Uses dynamically assembled, parameterized SQL. Article metadata, category, link, and distance constraints are combined with exact cosine ranking in one SQL statement.

The filtered implementation includes:

```sql
IGNORE INDEX (embedding)
```

This excludes the vector index from the filtered-search query, providing exact ranking over eligible chunks and avoiding selective-filter top-K under-fill.

#### Supported filters

| Filter | SQL behavior |
| --- | --- |
| `topics` | Match any specified topic, case-insensitively |
| `categories` | Match any specified Wikipedia category using relational SQL |
| `article_ids` | Include only specified article IDs |
| `exclude_article_ids` | Exclude specified article IDs |
| `min_length_chars` | Require `length_chars > min_length_chars` |
| `edited_since` | Require `revised_at >= edited_since` |
| `linked_from_article_id` | Restrict results to articles linked from the specified article |
| `max_distance` | Return only chunks within the specified cosine-distance threshold |

Conditions across different filter fields use **AND** semantics.

Multiple values within a topic, category, or included-ID field use **OR** semantics.

SQL values are passed using bound parameters rather than interpolated directly into query strings. Dynamically generated SQL fragments contain trusted query structure and placeholders.

#### Filter validation

The shared `SearchFilters` model provides:

```python
def validate(self) -> None:
```

The method validates all eight filter fields before retrieval.

Validation requirements include:

- `topics` and `categories` must be tuples containing non-empty strings.
- `article_ids` and `exclude_article_ids` must contain positive integer identifiers.
- `min_length_chars` must be a non-negative integer.
- `edited_since` must be a naive UTC `datetime`.
- `linked_from_article_id` must be a positive integer.
- `max_distance` must be a finite numeric value between 0 and 2.

Invalid values raise the project's shared `ValidationError`.

Both `WikiRepository` and `InMemoryRepository` call `SearchFilters.validate()` before executing searches with supplied filters.

This keeps filter-validation behavior consistent across production retrieval and fake-repository tests.

#### Verified behavior

Local tests cover:

- Every individual search-filter field.
- Multiple relational conditions in one SQL statement.
- Case-insensitive topic and category filtering.
- AND semantics across fields and OR semantics within multi-value fields.
- Article-length and cosine-distance boundaries.
- Filtering before top-K selection.
- Invalid filter inputs and boundary conditions.
- Fake-versus-MariaDB repository parity.

The latest reported local suite passed **273 tests: 182 unit tests and 91 integration tests**.

#### Limitations

Filtered searches currently use exact cosine ranking rather than indexed hybrid-search acceleration.

This design prioritizes correct filtered results over approximate nearest-neighbor speed.

HNSW-accelerated hybrid retrieval and its performance characteristics remain subjects for future experiments.

Recall and latency have not yet been benchmarked on a representative Wikipedia corpus.

### 6.5 Exact cosine baseline and HNSW tuning

Milestone 3 implements the two remaining database-search methods defined in the shared repository interface.

#### Exact cosine search

`WikiRepository.search_chunks_exact()` accepts the same arguments as `search_chunks()`:

```python
def search_chunks_exact(
    self,
    query_embedding: Sequence[float],
    *,
    top_k: int = 5,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
```

It uses MariaDB's native `VEC_DISTANCE_COSINE` function and explicitly excludes the `embedding` HNSW index through:

```sql
IGNORE INDEX (embedding)
```

Relational filters are applied within the SQL query before the final nearest-neighbor ranking and top-K limit.

The method provides an exact cosine-distance baseline for evaluating approximate HNSW results.

Integration tests verify:

- Correct cosine-distance ranking.
- Top-K limits.
- Returned article and chunk metadata.
- Support for hybrid SQL filters.
- Rejection of invalid inputs.
- Exclusion of the HNSW index in the `EXPLAIN` execution plan.

Both semantic and exact searches return the shared `SearchHit` data model.

For each result:

```python
score = 1.0 - distance
```

#### HNSW search breadth

`WikiRepository.set_ef_search(value)` configures MariaDB's `mhnsw_ef_search` session variable.

The method accepts integer values from 1 through 10000.

Invalid types and out-of-range values raise `ValidationError`.

The setting is changed using a parameterized statement:

```sql
SET SESSION mhnsw_ef_search = %s
```

Using `SET SESSION` limits the change to the repository's current MariaDB connection.

It does not modify the global server setting or another existing database session.

Integration tests verify:

- Accepted values, including supported boundaries.
- Rejection of invalid values.
- Persistence of the configured value on the current connection.
- Isolation between separate database connections.
- Unchanged global HNSW settings.

#### Remaining evaluation work

Successful implementation of `search_chunks_exact()` and `set_ef_search()` does not establish HNSW recall, latency, or an optimal search-breadth setting.

These conclusions require controlled experiments using the real Wikipedia corpus.

The evaluation should compare exact and approximate results for identical query embeddings, fixed top-K values, and documented HNSW settings.

Measurements should use repeated runs and report both retrieval quality and latency.

Results must be documented without presenting synthetic integration-test outcomes as Wikipedia retrieval benchmarks.

## 7. Verification and maintenance checklist

- The schema in `sql/001_schema.sql`, `verify_schema()`, repository SQL and this document must agree.
- Keep `EMBEDDING_DIM=384` and the model name synchronized across ingestion/query embedding generation and `embedding_meta`.
- Keep database integration tests isolated to `DB_NAME_TEST`; never recreate the main database during tests.
- Run `make test-unit`, `make test-db`, `make lint`, and `uv lock --check` before a database PR; inspect GitHub Actions results after pushing.
- Preserve signatures and return types in `docs/interfaces.md`, and add fake-versus-real repository contract tests before calling the repository implementation fully interchangeable.
- Record actual MariaDB version, HNSW parameters and experiment methodology with any published performance claims.
- Update README setup/reproduction instructions together with production features, and merge reviewed PRs into `main` because the official evaluator reads `main`, not unmerged branches.

