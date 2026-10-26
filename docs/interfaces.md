# WikiRAG — Shared Interfaces and Integration Contracts

**Project:** MariaDB Area 4 — Vector Search and RAG, Natively
**Status:** Proposed contract — requires review by all three contributors
**Reference:** MariaDB Area 4 Architecture Blueprint v0.3, Sections 7.1–7.9
**Authority:** The official hackathon brief governs the deliverables; this document governs internal integration once agreed.

## 1. Purpose

This document defines **the public interfaces shared between the three contributors**. It prevents one component from expecting a different function name, parameter, return type, data model, or error behavior than another component actually provides.

A signature shown here is a **target contract**, not proof that its implementation already exists. Before treating this document as frozen, compare it with the actual Python code, and have all contributors approve any differences.

## 2. Responsibilities and collaboration

| Contributor | Main area | Responsibilities |
| --- | --- | --- |
| **Database/integration contributor (you)** | `src/wikirag/db/`, foundation integration | MariaDB schema, transactional repository, database search, database tests, cross-module compatibility |
| **Zineb** | `src/wikirag/embeddings.py`, `src/wikirag/ingestion/` | Wikipedia acquisition, cleaning/chunking, real embeddings, ingestion pipeline |
| **Marwan** | `src/wikirag/retrieval/`, `src/wikirag/rag/`, `src/wikirag/evaluation/` | Retrieval, RAG, evaluation datasets, metrics, experiments |
| **All three contributors** | Shared files and integration decisions | Review interface changes, resolve integration problems, test PRs, keep `main` runnable |

**There is no assumed formal team-lead role.** File ownership identifies the primary implementer, not someone with unilateral authority to change the other contributors' interfaces. Any cross-module change must be reviewed with its affected contributors.

## 3. Global conventions

- Public methods use Python type hints and the names listed in this document.
- Shared immutable records come from `wikirag.models`; do not create competing versions.
- `None` means absent/unknown; `()` means an empty immutable collection. Do not use `""` as a missing-value sentinel.
- Times are **naive UTC `datetime`** values, consistently across modules.
- Wikipedia `article_id` is a positive integer. `chunk_index` starts at `0` and is contiguous within each article.
- `EMBEDDING_MODEL_NAME` and `EMBEDDING_DIM` are defined in `wikirag.config`:

  ```python
  EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
  EMBEDDING_DIM = 384
  ```

- Stored chunk embeddings use `tuple[float, ...]`; embedding functions return `list[float]`. Every vector has exactly **384 finite values**, and the embedder L2-normalizes its outputs.
- MariaDB VECTOR retrieval uses **cosine distance**: lower is better; `score = 1.0 - distance` (higher is better).
- Searches use **AND between filter fields**, **OR within a multivalued field**.
- A valid search with no matches returns `[]`, never an exception.
- Only project-level `WikiRagError` subclasses should cross component boundaries.

## 4. Shared models

**Source:** `src/wikirag/models.py`
**Shared responsibility:** Changes reviewed by every affected contributor.

### 4.1 Literals and ingestion records

```python
IngestOutcome = Literal["inserted", "updated", "skipped"]
RetrievalMode = Literal["semantic", "hybrid"]


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    article_id: int
    title: str
    url: str
    revision_id: int
    fingerprint: str
    revised_at: datetime
    length_chars: int
    fetched_at: datetime
    topic: str | None = None
    language: str = "en"
    categories: tuple[str, ...] = ()
    linked_titles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_index: int
    text: str
    embedding: tuple[float, ...]
    section_title: str | None = None


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: int
    article_id: int
    article_title: str
    chunk_index: int
    section_title: str | None
    text: str
```

`fingerprint` is a 64-character SHA-256 hexadecimal digest of the ingestion-relevant cleaned text and settings. `linked_titles` stores outgoing Wikipedia titles; only in-corpus targets are resolved to article IDs.

### 4.2 Search records

```python
@dataclass(frozen=True, slots=True)
class SearchFilters:
    topics: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    article_ids: tuple[int, ...] = ()
    exclude_article_ids: tuple[int, ...] = ()
    min_length_chars: int | None = None
    edited_since: datetime | None = None
    linked_from_article_id: int | None = None
    max_distance: float | None = None

    def is_empty(self) -> bool: ...
    def validate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: int
    article_id: int
    article_title: str
    article_url: str
    chunk_index: int
    section_title: str | None
    text: str
    distance: float
    score: float
```

| Filter | Meaning |
| --- | --- |
| `topics` | Match any given topic. |
| `categories` | Match any given category via the article/category association. |
| `article_ids` | Include only these article IDs. |
| `exclude_article_ids` | Exclude these article IDs. |
| `min_length_chars` | Strictly require `length_chars > min_length_chars`. |
| `edited_since` | Require `revised_at >= edited_since`. |
| `linked_from_article_id` | Include articles that the specified in-corpus article links to. |
| `max_distance` | Keep only results within the cosine-distance threshold. |

`SearchHit` is the public return type. Do **not** expose raw SQL tuples or dictionaries as search results.

`SearchFilters.validate()` raises `ValidationError` for invalid
filter values. Topic and category filters require tuples of
non-empty strings. Article IDs must be positive integers.
`min_length_chars` must be a non-negative integer.
`edited_since` must be a naive UTC datetime.
`max_distance` must be a finite number between 0 and 2.

Both `WikiRepository` and `InMemoryRepository` call this method
before executing a filtered search. This preserves consistent
validation across the real and fake implementations.

### 4.3 Pipeline, RAG, and evaluation records

```python
@dataclass(frozen=True, slots=True)
class IngestReport:
    inserted: int
    updated: int
    skipped: int
    failed: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    mode: RetrievalMode
    top_k: int
    filters: SearchFilters | None
    hits: tuple[SearchHit, ...]
    embed_ms: float
    search_ms: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: tuple[SearchHit, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    question: str
    answer: str
    context: ContextBundle
    retrieval: RetrievalResult
    generator_name: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    text: str
    relevant_article_ids: frozenset[int]
    filters: SearchFilters | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    n_queries: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    latency_ms_p50: float
    latency_ms_p95: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    embedding_model: str
    corpus_articles: int
    corpus_chunks: int
    ks: tuple[int, ...]
    modes: dict[str, ModeMetrics]
    per_query: tuple[dict, ...]
```

`IngestReport.failed` contains `(article_title, error_message)` pairs. `EvalQuery.relevant_article_ids` must be nonempty. `ContextBundle.sources` correspond to `[1]`, `[2]`, etc. in `ContextBundle.text`.

## 5. Database interface — Database/integration contributor

**Source:** `src/wikirag/db/repository.py`
**Class:** `WikiRepository`

The repository is the application-facing database API. Ingestion and retrieval call its methods instead of writing SQL themselves.

```python
class WikiRepository:
    def __init__(self, config: DbConfig | None = None) -> None: ...
    def close(self) -> None: ...

    def ensure_embedding_meta(self, model_name: str, dim: int) -> None: ...

    def ingest_article(
        self,
        article: ArticleRecord,
        chunks: Sequence[ChunkRecord],
    ) -> IngestOutcome: ...

    def get_fingerprint(self, article_id: int) -> str | None: ...
    def resolve_links(self) -> int: ...
    def delete_article(self, article_id: int) -> bool: ...

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...

    def get_chunk(self, chunk_id: int) -> StoredChunk | None: ...

    def get_neighbor_chunks(
        self,
        chunk_id: int,
        *,
        window: int = 1,
    ) -> list[StoredChunk]: ...

    def get_article(self, article_id: int) -> ArticleRecord | None: ...
    def get_article_by_title(self, title: str) -> ArticleRecord | None: ...
    def get_linked_articles(self, article_id: int) -> list[ArticleRecord]: ...
    def list_topics(self) -> list[str]: ...
    def list_categories(self, *, min_articles: int = 1) -> list[str]: ...
    def count_articles(self) -> int: ...
    def count_chunks(self) -> int: ...

    def search_chunks_exact(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...

    def set_ef_search(self, value: int) -> None: ...
```

### 5.1 Lifecycle and metadata

- When `config is None`, use `load_settings().db`.
- Support `with WikiRepository(...) as repo:` as well as explicit `close()`.
- `ensure_embedding_meta()` stores metadata initially, accepts identical values, and raises `SchemaMismatchError` for incompatible model/dimensions.

### 5.2 Atomic, idempotent ingestion

1. Validate the article, chunks, chunk ordering, and embedding dimensions **before** beginning mutation.
2. Begin a transaction and read the current fingerprint.
3. If unchanged, commit without writes and return `"skipped"`.
4. Otherwise upsert article metadata and replace its chunks, category associations, and outgoing link rows atomically.
5. Commit and return `"inserted"` or `"updated"`.
6. On database failure, **roll back** and raise `DatabaseError`; do not leave partially replaced data.

`resolve_links()` runs once after ingestion, resolves links only to in-corpus articles, and returns the number resolved. `delete_article()` returns `False` if the article does not exist; otherwise it removes dependent records.

Article title lookup and link resolution use exact, case-sensitive title matching, consistent with the database's binary title collation. The ingestion pipeline must normalize Wikipedia titles consistently before storing articles and outgoing links.

### 5.3 Search semantics

- Validate `top_k` (integer from 1 to 100) and query vector.
- `search_chunks()` uses MariaDB's `VECTOR` column, vector index, and `VEC_DISTANCE_COSINE` in native SQL.
- Return results ordered by ascending `distance`, with `score = 1 - distance`.
- Apply ordinary SQL predicates/joins for `SearchFilters` in conjunction with vector ranking.
- Return `min(top_k, eligible_matching_chunks)` hits, including under selective filters; avoid silent HNSW under-fill.
- `max_distance` can remove results but cannot add results.
- `search_chunks_exact()` computes exact top-K cosine results without relying on the HNSW index; used in evaluation.
- `set_ef_search()` sets the positive `mhnsw_ef_search` value for the current session.
- No matches return `[]`. Missing `get_*` records return `None` or `[]` as typed.
- `get_neighbor_chunks()` includes the central chunk, stays within its article, and preserves `chunk_index` order.

### 5.4 Fake implementation

`tests/fakes.py` defines `InMemoryRepository` with **the same public methods and observable behavior**, but using Python memory and brute-force cosine distance. It must pass shared repository contract tests alongside the real implementation.

## 6. Embedding interface — Zineb

**Source:** `src/wikirag/embeddings.py`

```python
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None: ...


def get_embedder() -> Embedder: ...
```

**Rules:** An empty input list returns `[]`; any blank string raises `ValidationError`; output preserves input order; vectors have dimension 384, finite values, and L2 norm approximately 1; identical texts produce consistent vectors. `get_embedder()` caches the model process-wide. `tests/fakes.py::FakeEmbedder` follows the same embedding protocol but produces deterministic **nonsemantic** vectors.

## 7. Ingestion interface — Zineb

**Primary source:** `src/wikirag/ingestion/pipeline.py`

```python
def compute_fingerprint(
    cleaned_text: str,
    *,
    max_chars: int,
    overlap_chars: int,
    model_name: str,
) -> str: ...


def ingest_snapshot(
    snapshot_path: Path,
    repo: WikiRepository,
    embedder: Embedder,
    *,
    max_chars: int = 1000,
    overlap_chars: int = 150,
    limit: int | None = None,
) -> IngestReport: ...
```

**Wikipedia helper source:** `src/wikirag/ingestion/wikipedia.py`

```python
def load_seed_titles(path: Path) -> list[tuple[str, str]]: ...


def build_snapshot(
    seed_path: Path,
    out_path: Path,
    *,
    delay_s: float = 0.2,
) -> int: ...


def load_snapshot(path: Path) -> Iterator[RawArticle]: ...
```

`RawArticle` is internal to ingestion and must be defined consistently within that module.

**Required pipeline order:**

1. Call `repo.ensure_embedding_meta(embedder.model_name, embedder.dim)`.
2. Load and clean a Wikipedia article; compute its fingerprint.
3. Call `repo.get_fingerprint(article_id)` **before embedding** to avoid work on unchanged articles.
4. Chunk changed articles; call `embedder.embed_texts()`; build `ArticleRecord` and `ChunkRecord` values.
5. Call `repo.ingest_article()` and count the outcome.
6. Record per-article failures in `IngestReport.failed`, then continue with other articles.
7. Call `repo.resolve_links()` **once** after the ingestion run.

Ingestion must not open a competing database connection or issue SQL itself.

## 8. Retrieval interface — Marwan

**Source:** `src/wikirag/retrieval/retriever.py`

```python
class Retriever:
    def __init__(self, repo: WikiRepository, embedder: Embedder) -> None: ...

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = "semantic",
        filters: SearchFilters | None = None,
    ) -> RetrievalResult: ...
```

- Constructor checks repository/model compatibility via `repo.ensure_embedding_meta()`.
- Blank query, unknown mode, or invalid `top_k` raises `ValidationError`.
- `"semantic"` **ignores** filters.
- `"hybrid"` requires nonempty filters (`SearchFilters.is_empty()` must be `False`); otherwise raise `ValidationError`.
- Embed with `embedder.embed_query()`, then call `repo.search_chunks()`.
- Return `RetrievalResult` with `hits` as a **tuple**, even though the repository returns a list.
- Capture `embed_ms`, `search_ms`, and overall `latency_ms` separately.
- No results mean `hits=()`.

Example:

```python
result = retriever.retrieve(
    "How do black holes form?",
    top_k=5,
    mode="hybrid",
    filters=SearchFilters(topics=("physics",)),
)
```

## 9. RAG interfaces — Marwan

**Context builder:** `src/wikirag/rag/context.py`

```python
def build_context(
    hits: Sequence[SearchHit],
    *,
    max_chars: int = 6000,
) -> ContextBundle: ...
```

Preserve hit order, remove duplicate `chunk_id`s, number citations `[1]`, `[2]`, etc., never cut a chunk in half, and set `truncated=True` if chunks are omitted.

**Generator:** `src/wikirag/rag/generator.py`

```python
class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, question: str, context: ContextBundle) -> str: ...
```

**RAG pipeline:** `src/wikirag/rag/pipeline.py`

```python
def answer_question(
    question: str,
    retriever: Retriever,
    generator: Generator,
    *,
    top_k: int = 5,
    mode: RetrievalMode = "semantic",
    filters: SearchFilters | None = None,
    max_context_chars: int = 6000,
) -> RAGAnswer: ...
```

On zero retrieval hits, return a fixed no-information answer **without calling the generator**. Successful generated answers must reference numbered sources; generator failures must become project-level `WikiRagError` exceptions.

## 10. Evaluation interfaces — Marwan

**Metrics:** `src/wikirag/evaluation/metrics.py`

```python
def recall_at_k(retrieved: Sequence[int], relevant: Collection[int], k: int) -> float: ...


def precision_at_k(retrieved: Sequence[int], relevant: Collection[int], k: int) -> float: ...


def reciprocal_rank(retrieved: Sequence[int], relevant: Collection[int]) -> float: ...


def mean(values: Sequence[float]) -> float: ...
```

**Dataset:** `src/wikirag/evaluation/dataset.py`

```python
def load_queries(path: Path) -> list[EvalQuery]: ...
```

**Runner:** `src/wikirag/evaluation/run_eval.py`

```python
def run_evaluation(
    queries: Sequence[EvalQuery],
    retriever: Retriever,
    *,
    modes: Sequence[RetrievalMode] = ("semantic", "hybrid"),
    ks: Sequence[int] = (1, 3, 5, 10),
    chunk_pool: int = 30,
    repeats: int = 5,
) -> EvalReport: ...


def run_rag_smoke_check(
    questions: Sequence[str],
    retriever: Retriever,
    generator: Generator,
    *,
    top_k: int = 5,
) -> list[RAGAnswer]: ...


def write_report(report: EvalReport, json_path: Path, md_path: Path) -> None: ...
```

Relevance is assessed at the **article** level, not the chunk level. Deduplicate ranked chunk hits to unique article IDs before metric calculations. Precision@K divides by K. Invalid K or empty relevant sets raise `ValidationError`. Evaluate hybrid only for questions with filters; compare semantic on the same subset. Warm up before repeated latency measurement and report p50/p95.

## 11. Shared errors

**Source:** `src/wikirag/errors.py`

| Exception | Contract |
| --- | --- |
| `WikiRagError` | Base project exception. |
| `ValidationError` | Invalid article, text, query, filter, mode, or limit. |
| `EmbeddingDimensionError` | Wrong dimension or invalid vector; **inherits `ValidationError`**. |
| `DatabaseConnectionError` | Cannot connect or connection was lost; never include passwords. |
| `DatabaseError` | SQL/transaction failure; rollback required. |
| `SchemaMismatchError` | Missing schema or incompatible embedding metadata. |

**Rule:** Validate before mutation; do not report success after a failed transaction. Translate third-party exceptions into appropriate project exceptions at module boundaries.

## 12. Import and integration rules

```text
wikirag.config / wikirag.errors / wikirag.models
                           |
             +-------------+--------------+
             |                            |
      embeddings.py                   db/repository.py
             |                            |
             +-------- ingestion/ --------+
             |                            |
             +-------- retrieval/ --------+
                                          |
                                        rag/
                                          |
                                      evaluation/

cli.py composes the components. tests/fakes.py supports parallel tests.
```

- Only `db/` owns application SQL; avoid direct SQL in ingestion, retrieval, and RAG.
- No duplicate definitions of shared models.
- Do not change public APIs without updating all affected callers, fakes, documentation, and tests.
- Before merging any component, run lint and tests, validate imports and signatures, and request review from contributors whose modules depend on the change.
- **Work on feature branches**, open pull requests, and keep `main` runnable. Evaluators inspect `main`, so accepted work must be merged there before review deadlines.
- `InMemoryRepository` and `WikiRepository` must pass the same repository contract tests wherever feasible.

## 13. Interface-change protocol

1. Open a PR explaining the proposed change and which modules/callers are affected.
2. Discuss it with **all affected contributors**. No single contributor unilaterally changes shared contracts.
3. Update implementation signatures, `models.py` if needed, **this file**, `tests/fakes.py` when relevant, and the contract tests in the **same PR**.
4. Run formatting, lint, unit tests, and applicable integration tests.
5. Obtain agreement from the contributors whose code consumes or provides the modified interface, then merge.
6. AI coding assistants must **not silently rename public functions, fields, types, or defaults**.

## 14. Freeze checklist

Before the team treats this document as final:

- [ ] Compare the signatures and dataclass fields here with actual source files.
- [ ] Confirm the `InMemoryRepository` fake implements the same repository contract.
- [ ] Zineb reviews the embedding and ingestion APIs.
- [ ] Marwan reviews retrieval, RAG, and evaluation APIs.
- [ ] The database/integration contributor reviews the repository APIs and SQL obligations.
- [ ] All three agree on any remaining changes.
- [ ] Commit the approved contract on a feature branch and merge through a reviewed PR.

**Until approval, this is a proposed integration contract, not a statement that every module is already implemented or reviewed.**
