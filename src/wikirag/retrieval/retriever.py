"""Query embedding and database-backed WikiRAG retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from wikirag.errors import ValidationError
from wikirag.models import (
    RetrievalMode,
    RetrievalResult,
    SearchFilters,
    SearchHit,
)


class RetrieverEmbedder(Protocol):
    """Temporary narrow interface until the shared embedder interface lands."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]: ...


class RetrieverRepository(Protocol):
    """Temporary narrow interface until the shared repository interface lands."""

    def ensure_embedding_meta(self, model_name: str, dim: int) -> None: ...

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...


class Retriever:
    """Retrieve ranked article chunks using semantic or hybrid search."""

    def __init__(self, repo: RetrieverRepository, embedder: RetrieverEmbedder) -> None:
        """Initialize retrieval and validate embedding compatibility.

        Args:
            repo: Repository used to validate metadata and search chunks.
            embedder: Embedder used to create query vectors.

        Raises:
            SchemaMismatchError: If stored embedding metadata is incompatible.
        """
        self._repo = repo
        self._embedder = embedder

        # Fail early if the stored vectors use an incompatible model.
        self._repo.ensure_embedding_meta(
            self._embedder.model_name,
            self._embedder.dim,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = "semantic",
        filters: SearchFilters | None = None,
    ) -> RetrievalResult:
        """Embed a query, search the repository, and return ranked hits.

        Args:
            query: Non-empty user query.
            top_k: Maximum number of hits to return, between 1 and 100.
            mode: ``semantic`` ignores filters; ``hybrid`` applies them.
            filters: Optional search constraints used only in hybrid mode.

        Returns:
            RetrievalResult containing ranked hits and timing measurements.

        Raises:
            ValidationError: If the query, limit, mode, or hybrid filters are invalid.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValidationError("Query must be a non-empty string")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValidationError("top_k must be an integer between 1 and 100")

        if mode not in ("semantic", "hybrid"):
            raise ValidationError("mode must be 'semantic' or 'hybrid'")

        if mode == "hybrid" and (filters is None or filters.is_empty()):
            raise ValidationError("Hybrid retrieval requires at least one active filter")

        # Semantic mode deliberately ignores filters.
        applied_filters = filters if mode == "hybrid" else None

        started = perf_counter()

        embed_started = perf_counter()
        query_embedding = self._embedder.embed_query(query)
        embed_ms = (perf_counter() - embed_started) * 1000

        search_started = perf_counter()
        hits = self._repo.search_chunks(
            query_embedding,
            top_k=top_k,
            filters=applied_filters,
        )
        search_ms = (perf_counter() - search_started) * 1000

        latency_ms = (perf_counter() - started) * 1000

        return RetrievalResult(
            query=query,
            mode=mode,
            top_k=top_k,
            filters=applied_filters,
            hits=tuple(hits),
            embed_ms=embed_ms,
            search_ms=search_ms,
            latency_ms=latency_ms,
        )
