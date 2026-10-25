# Retrieval and RAG

## Retrieval behavior

`Retriever` embeds a user query and asks the repository to return ranked
chunk matches. It returns a `RetrievalResult` containing the query, selected
mode, effective filters, hits, and timing measurements.

- **Semantic retrieval** searches by the query embedding. Any supplied filters
  are deliberately ignored.
- **Hybrid retrieval** combines vector search with repository-supported
  filters. At least one active filter is required.

The retriever accepts `top_k` values from 1 through 100. Queries must be
non-empty strings, and the mode must be `semantic` or `hybrid`.

## Embedding compatibility

When a `Retriever` is constructed, it calls
`ensure_embedding_meta()` with the embedder's model name and vector dimension.
The repository is responsible for rejecting metadata incompatible with its
stored embeddings. This check fails early, before query retrieval.

## Timing measurements

`RetrievalResult` reports timings in milliseconds:

- `embed_ms`: time spent embedding the query.
- `search_ms`: time spent searching the repository.
- `latency_ms`: total time measured around embedding and search.

These timings help identify whether query embedding or repository search
dominates retrieval latency.

## Interfaces

The retriever currently declares narrow structural protocols named
`RetrieverEmbedder` and `RetrieverRepository`. They describe only the
operations needed by this component and are temporary until the shared
production interfaces are available. Coordinate any interface consolidation
with the team; do not independently change agreed shared signatures.

## Tests and database integration

Unit tests use `FakeEmbedder` and `InMemoryRepository` to exercise retrieval
without requiring a running database. They cover validation, timing fields,
filter behavior, returned hits, and embedding metadata incompatibility.

After the production database repository and vector-search implementation
are integrated, add an integration test against MariaDB to verify real
semantic search, hybrid filtering, result ranking, and embedding metadata
compatibility. The in-memory tests do not replace that database integration
test.
