"""
Reusable fake components for the WikiRAG test suite.

FakeEmbedder produces deterministic vectors without loading an AI model.
InMemoryRepository will simulate MariaDB operations in a later step.

These components are intended for testing, not production retrieval.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from wikirag.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, DbConfig
from wikirag.errors import (
    DatabaseConnectionError,
    EmbeddingDimensionError,
    SchemaMismatchError,
    ValidationError,
)
from wikirag.models import (
    ArticleRecord,
    ChunkRecord,
    IngestOutcome,
    SearchFilters,
    SearchHit,
    StoredChunk,
)


class FakeEmbedder:
    """
    Generate deterministic embeddings for unit and integration-style tests.

    The class follows the public Embedder interface defined in Claude's
    architecture blueprint.

    Hash-based vectors are reproducible and L2-normalized, but they
    do not represent actual semantic similarity.

    Attributes:
        model_name (str): Identifier of the simulated embedding model.
        dim (int): Number of dimensions in each embedding vector.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        """
        Initialize the fake embedding model.

        Args:
            model_name (str): Model identifier used when generating vectors.

        Returns:
            None: Initializes the fake embedder.

        Raises:
            ValidationError: If model_name is not a non-empty string.
        """

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Model name must be a non-empty string")

        self._model_name = model_name.strip()

    @property
    def model_name(self) -> str:
        """
        Return the simulated model identifier.

        Returns:
            str: Name associated with this fake embedding model.
        """

        return self._model_name

    @property
    def dim(self) -> int:
        """
        Return the expected embedding dimension.

        Returns:
            int: Number of embedding dimensions, currently 384.
        """

        return EMBEDDING_DIM

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for a sequence of text strings.

        Args:
            texts (Sequence[str]): Text strings to embed.
            batch_size (int): Maximum processing batch size. Accepted
                for compatibility with the real Embedder interface.

        Returns:
            list[list[float]]: One normalized vector per input string,
                in the same order. Returns [] for an empty sequence.

        Raises:
            ValidationError: If batch_size is not a positive integer,
                texts is not a sequence, or any text is blank.
        """

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValidationError("batch_size must be a positive integer")

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValidationError("texts must be a sequence of strings")

        # Reject the entire batch if any text is invalid.
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValidationError("Every input text must be a non-empty string")

        # The fake does not need batching but preserves the real interface.
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """
        Generate one deterministic query embedding.

        Args:
            text (str): Non-empty query to embed.

        Returns:
            list[float]: Normalized embedding containing 384 values.

        Raises:
            ValidationError: If text is not a non-empty string.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Query must be a non-empty string")

        # Use the same mapping for documents and queries.
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """
        Convert validated text into a deterministic numeric vector.

        SHAKE-256 generates reproducible bytes independently of Python's
        randomized hash() behavior.

        Args:
            text (str): Previously validated text to convert.

        Returns:
            list[float]: A finite, L2-normalized vector of EMBEDDING_DIM
                floating-point values.

        Raises:
            ValidationError: If the generated vector cannot be normalized.
        """

        # Different model identifiers should produce different vectors.
        payload = f"{self._model_name}\0{text}".encode()

        # Generate four bytes for each vector component.
        raw_bytes = hashlib.shake_256(payload).digest(EMBEDDING_DIM * 4)

        values: list[float] = []

        for index in range(EMBEDDING_DIM):
            start = index * 4
            component_bytes = raw_bytes[start : start + 4]

            integer_value = int.from_bytes(
                component_bytes,
                byteorder="big",
            )

            # Convert the unsigned integer into a value near [-1, 1].
            value = (integer_value / 2**32) * 2.0 - 1.0
            values.append(value)

        # L2 normalization ensures cosine comparisons are well-defined.
        norm = math.sqrt(math.fsum(value * value for value in values))

        if not math.isfinite(norm) or norm == 0:
            raise ValidationError("Generated embedding cannot be normalized")

        return [value / norm for value in values]


class InMemoryRepository:
    """
    Simulate the WikiRepository interface without a MariaDB server.

    Articles, chunks, embeddings, and resolved links are held in memory.
    Search uses exact cosine distance instead of an HNSW vector index.

    The fake follows the real repository's public method signatures so
    ingestion and retrieval components can use either implementation.

    Attributes:
        _articles (dict[int, ArticleRecord]): Stored article metadata.
        _chunks (dict[int, StoredChunk]): Stored chunk records.
        _vectors (dict[int, tuple[float, ...]]): Embeddings by chunk ID.
        _links (dict[int, set[int]]): Resolved article-to-article links.
    """

    def __init__(self, config: DbConfig | None = None) -> None:
        """
        Initialize an empty in-memory repository.

        Args:
            config (DbConfig | None): Optional database configuration.
                Accepted for compatibility with WikiRepository but unused.

        Returns:
            None: Initializes the repository.
        """

        # No database connection or credentials are needed by this fake.
        self._articles: dict[int, ArticleRecord] = {}
        self._chunks: dict[int, StoredChunk] = {}
        self._vectors: dict[int, tuple[float, ...]] = {}
        self._links: dict[int, set[int]] = {}

        self._next_chunk_id = 1
        self._embedding_meta: tuple[str, int] | None = None
        self._closed = False
        self._ef_search = 20

    def _check_open(self) -> None:
        """
        Ensure the repository has not been closed.

        Returns:
            None: The repository is available.

        Raises:
            DatabaseConnectionError: If the repository is closed.
        """

        if self._closed:
            raise DatabaseConnectionError("Repository is closed")

    def __enter__(self) -> Self:
        """
        Enter the repository context manager.

        Returns:
            InMemoryRepository: The current repository instance.

        Raises:
            DatabaseConnectionError: If the repository is closed.
        """

        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close the repository when leaving a context manager.

        Args:
            exc_type (type[BaseException] | None): Exception type, if any.
            exc_value (BaseException | None): Exception instance, if any.
            traceback (TracebackType | None): Exception traceback, if any.

        Returns:
            None: Closes the repository and does not suppress exceptions.
        """

        self.close()

    def close(self) -> None:
        """
        Mark the repository as closed.

        Returns:
            None: Subsequent repository operations are rejected.
        """

        self._closed = True

    def ensure_embedding_meta(self, model_name: str, dim: int) -> None:
        """
        Check that the embedding model matches stored metadata.

        Args:
            model_name (str): Identifier of the embedding model.
            dim (int): Expected embedding dimension.

        Returns:
            None: Stores metadata initially or confirms compatibility.

        Raises:
            ValidationError: If the model name is blank.
            SchemaMismatchError: If the dimension is not 384 or the
                supplied metadata conflicts with existing metadata.
        """

        self._check_open()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Model name cannot be blank")

        if dim != EMBEDDING_DIM:
            raise SchemaMismatchError(f"Expected embedding dimension {EMBEDDING_DIM}, got {dim}")

        requested_meta = (model_name.strip(), dim)

        if self._embedding_meta is None:
            self._embedding_meta = requested_meta
        elif self._embedding_meta != requested_meta:
            raise SchemaMismatchError("Embedding model or dimension does not match stored metadata")

    @staticmethod
    def _validate_vector(
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        """
        Validate an embedding before storage or searching.

        Args:
            embedding (Sequence[float]): Numeric embedding to validate.

        Returns:
            tuple[float, ...]: Validated, finite embedding values.

        Raises:
            EmbeddingDimensionError: If the vector has an incorrect
                dimension, non-numeric values, or zero magnitude.
        """

        if isinstance(embedding, (str, bytes)):
            raise EmbeddingDimensionError("Embedding must be numeric")

        try:
            values = tuple(float(value) for value in embedding)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingDimensionError("Embedding must contain numeric values") from exc

        if len(values) != EMBEDDING_DIM:
            raise EmbeddingDimensionError(f"Expected {EMBEDDING_DIM} dimensions, got {len(values)}")

        if not all(math.isfinite(value) for value in values):
            raise EmbeddingDimensionError("Embedding contains NaN or infinite values")

        if not any(value != 0.0 for value in values):
            raise EmbeddingDimensionError("Embedding cannot be a zero vector")

        return values

    def ingest_article(
        self,
        article: ArticleRecord,
        chunks: Sequence[ChunkRecord],
    ) -> IngestOutcome:
        """
        Insert, update, or skip an article and its chunks.

        All input is validated before changing stored records.
        An article with an unchanged fingerprint is skipped.

        Args:
            article (ArticleRecord): Article metadata to store.
            chunks (Sequence[ChunkRecord]): Ordered text chunks and
                their embeddings.

        Returns:
            IngestOutcome: "inserted", "updated", or "skipped".

        Raises:
            ValidationError: If article metadata, chunk text, or
                chunk ordering is invalid.
            EmbeddingDimensionError: If a chunk embedding is invalid.
            DatabaseConnectionError: If the repository is closed.
        """

        self._check_open()

        if not isinstance(article, ArticleRecord):
            raise ValidationError("Expected an ArticleRecord")

        if type(article.article_id) is not int or article.article_id <= 0:
            raise ValidationError("article_id must be positive")

        if not article.title.strip() or not article.url.strip():
            raise ValidationError("Article title and URL cannot be blank")

        if article.length_chars < 0:
            raise ValidationError("Article length cannot be negative")

        # Fingerprints are SHA-256 digests represented as hexadecimal text.
        fingerprint = article.fingerprint
        if len(fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in fingerprint.lower()
        ):
            raise ValidationError("Fingerprint must be a 64-character SHA-256 hex digest")

        if not chunks:
            raise ValidationError("An article must contain at least one chunk")

        validated_chunks: list[tuple[ChunkRecord, tuple[float, ...]]] = []

        for expected_index, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise ValidationError("Expected a ChunkRecord")

            if chunk.chunk_index != expected_index:
                raise ValidationError("Chunk indices must start at zero and be contiguous")

            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ValidationError("Chunk text cannot be blank")

            # Validate everything before mutating existing article data.
            vector = self._validate_vector(chunk.embedding)
            validated_chunks.append((chunk, vector))

        existing = self._articles.get(article.article_id)

        if existing is not None and existing.fingerprint == fingerprint:
            return "skipped"

        outcome: IngestOutcome = "updated" if existing else "inserted"

        # Remove old chunks only after the replacement has been validated.
        old_ids = [
            chunk_id
            for chunk_id, stored in self._chunks.items()
            if stored.article_id == article.article_id
        ]

        for chunk_id in old_ids:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]

        self._articles[article.article_id] = article

        for chunk, vector in validated_chunks:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1

            self._chunks[chunk_id] = StoredChunk(
                chunk_id=chunk_id,
                article_id=article.article_id,
                article_title=article.title,
                chunk_index=chunk.chunk_index,
                section_title=chunk.section_title,
                text=chunk.text,
            )
            self._vectors[chunk_id] = vector

        # Links are resolved separately after the ingestion run.
        self._links.pop(article.article_id, None)

        return outcome

    def get_fingerprint(self, article_id: int) -> str | None:
        """
        Return the stored fingerprint of an article.

        Args:
            article_id (int): Wikipedia article identifier.

        Returns:
            str | None: Stored fingerprint or None if the article is absent.
        """

        self._check_open()
        article = self._articles.get(article_id)
        return article.fingerprint if article else None

    def resolve_links(self) -> int:
        """
        Resolve stored article titles into in-corpus article links.

        Links pointing to articles outside the stored corpus are ignored.

        Returns:
            int: Number of resolved article-to-article links.
        """

        self._check_open()

        # Match Wikipedia titles exactly, as in the MariaDB repository.
        title_to_id = {article.title: article.article_id for article in self._articles.values()}

        resolved: dict[int, set[int]] = {}

        for article in self._articles.values():
            targets: set[int] = set()

            for linked_title in article.linked_titles:
                target_id = title_to_id.get(linked_title)

                if target_id is not None:
                    targets.add(target_id)

            resolved[article.article_id] = targets

        self._links = resolved
        return sum(len(targets) for targets in resolved.values())

    def delete_article(self, article_id: int) -> bool:
        """
        Delete an article and its dependent chunks and links.

        Args:
            article_id (int): Article identifier to remove.

        Returns:
            bool: True if the article existed; otherwise False.
        """

        self._check_open()

        if article_id not in self._articles:
            return False

        del self._articles[article_id]

        chunk_ids = [
            chunk_id for chunk_id, chunk in self._chunks.items() if chunk.article_id == article_id
        ]

        for chunk_id in chunk_ids:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]

        # Simulate ON DELETE CASCADE for inbound and outbound links.
        self._links.pop(article_id, None)

        for target_ids in self._links.values():
            target_ids.discard(article_id)

        return True

    @staticmethod
    def _cosine_distance(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        """
        Calculate cosine distance between two validated vectors.

        Args:
            first (Sequence[float]): First numeric vector.
            second (Sequence[float]): Second numeric vector.

        Returns:
            float: Cosine distance between 0 and 2.
        """

        dot_product = math.fsum(a * b for a, b in zip(first, second))
        first_norm = math.sqrt(math.fsum(value * value for value in first))
        second_norm = math.sqrt(math.fsum(value * value for value in second))

        distance = 1.0 - dot_product / (first_norm * second_norm)

        # Clamp minor floating-point rounding outside the theoretical range.
        return max(0.0, min(2.0, distance))

    def _matches_filters(
        self,
        article: ArticleRecord,
        filters: SearchFilters,
    ) -> bool:
        """
        Check whether an article satisfies relational search filters.

        Args:
            article (ArticleRecord): Article to evaluate.
            filters (SearchFilters): Active hybrid-search constraints.

        Returns:
            bool: True if the article satisfies every active constraint.
        """

        if filters.topics and (
            article.topic is None
            or article.topic.casefold() not in {topic.casefold() for topic in filters.topics}
        ):
            return False

        if filters.categories:
            allowed = {category.casefold() for category in filters.categories}
            article_categories = {category.casefold() for category in article.categories}

            if not allowed.intersection(article_categories):
                return False

        if filters.article_ids and article.article_id not in filters.article_ids:
            return False

        if article.article_id in filters.exclude_article_ids:
            return False

        # The blueprint specifies a strict greater-than length predicate.
        if (
            filters.min_length_chars is not None
            and article.length_chars <= filters.min_length_chars
        ):
            return False

        if filters.edited_since is not None and article.revised_at < filters.edited_since:
            return False

        if filters.linked_from_article_id is not None:
            linked_ids = self._links.get(filters.linked_from_article_id, set())

            if article.article_id not in linked_ids:
                return False

        return True

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """
        Search stored chunks using exact cosine distance and SQL-like filters.

        Filters are combined using AND across fields and OR within
        multi-valued fields. Results are sorted by increasing distance.

        Args:
            query_embedding (Sequence[float]): Query vector.
            top_k (int): Maximum number of results, from 1 through 100.
            filters (SearchFilters | None): Optional hybrid-search filters.

        Returns:
            list[SearchHit]: Ranked results satisfying all active filters.

        Raises:
            ValidationError: If top_k or filters are invalid.
            EmbeddingDimensionError: If the query vector is invalid.
            DatabaseConnectionError: If the repository is closed.
        """

        self._check_open()

        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValidationError("top_k must be between 1 and 100")

        if filters is not None:
            if not isinstance(filters, SearchFilters):
                raise ValidationError("filters must be SearchFilters or None")

            filters.validate()

        query_vector = self._validate_vector(query_embedding)
        ranked: list[SearchHit] = []

        for chunk_id, chunk in self._chunks.items():
            article = self._articles[chunk.article_id]

            if filters is not None and not self._matches_filters(article, filters):
                continue

            distance = self._cosine_distance(
                query_vector,
                self._vectors[chunk_id],
            )

            if (
                filters is not None
                and filters.max_distance is not None
                and distance > filters.max_distance
            ):
                continue

            ranked.append(
                SearchHit(
                    chunk_id=chunk_id,
                    article_id=article.article_id,
                    article_title=article.title,
                    article_url=article.url,
                    chunk_index=chunk.chunk_index,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    distance=distance,
                    score=1.0 - distance,
                )
            )

        # Use deterministic tie-breakers to keep unit tests reproducible.
        ranked.sort(
            key=lambda hit: (
                hit.distance,
                hit.article_id,
                hit.chunk_index,
                hit.chunk_id,
            )
        )

        return ranked[:top_k]

    def search_chunks_exact(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """
        Perform exact vector search for evaluation and comparison.

        The fake repository always uses exact search, so this method
        delegates to search_chunks.

        Args:
            query_embedding (Sequence[float]): Query vector.
            top_k (int): Maximum number of results.
            filters (SearchFilters | None): Optional relational filters.

        Returns:
            list[SearchHit]: Exact ranked search results.

        Raises:
            ValidationError: If search arguments are invalid.
            EmbeddingDimensionError: If the query vector is invalid.
            DatabaseConnectionError: If the repository is closed.
        """

        return self.search_chunks(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )

    def get_chunk(self, chunk_id: int) -> StoredChunk | None:
        """
        Retrieve a stored chunk by its database-style identifier.

        Args:
            chunk_id (int): Identifier of the requested chunk.

        Returns:
            StoredChunk | None: Matching chunk or None when absent.
        """

        self._check_open()
        return self._chunks.get(chunk_id)

    def get_neighbor_chunks(
        self,
        chunk_id: int,
        *,
        window: int = 1,
    ) -> list[StoredChunk]:
        """
        Return neighboring chunks from the same article.

        Args:
            chunk_id (int): Identifier of the central chunk.
            window (int): Number of chunks to include on either side.

        Returns:
            list[StoredChunk]: Ordered neighboring chunks, including
                the central chunk. Returns [] if the chunk is absent.

        Raises:
            ValidationError: If window is negative or non-integer.
        """

        self._check_open()

        if type(window) is not int or window < 0:
            raise ValidationError("window must be a non-negative integer")

        center = self._chunks.get(chunk_id)

        if center is None:
            return []

        neighbors = [
            chunk
            for chunk in self._chunks.values()
            if chunk.article_id == center.article_id
            and abs(chunk.chunk_index - center.chunk_index) <= window
        ]

        return sorted(neighbors, key=lambda chunk: chunk.chunk_index)

    def get_article(self, article_id: int) -> ArticleRecord | None:
        """
        Retrieve article metadata by Wikipedia identifier.

        Args:
            article_id (int): Wikipedia article identifier.

        Returns:
            ArticleRecord | None: Matching article or None when absent.
        """

        self._check_open()
        return self._articles.get(article_id)

    def get_article_by_title(self, title: str) -> ArticleRecord | None:
        """
        Retrieve an article using its title.

        Args:
            title (str): Wikipedia article title to match.

        Returns:
            ArticleRecord | None: Matching article or None when absent.
        """

        self._check_open()

        for article in self._articles.values():
            if article.title == title:
                return article

        return None

    def get_linked_articles(self, article_id: int) -> list[ArticleRecord]:
        """
        Return in-corpus articles linked from a particular article.

        Args:
            article_id (int): Identifier of the source article.

        Returns:
            list[ArticleRecord]: Resolved target articles ordered by ID.
        """

        self._check_open()

        target_ids = sorted(self._links.get(article_id, set()))

        return [
            self._articles[target_id] for target_id in target_ids if target_id in self._articles
        ]

    def list_topics(self) -> list[str]:
        """
        List distinct non-empty corpus topics.

        Returns:
            list[str]: Sorted topic names.
        """

        self._check_open()

        topics = {article.topic for article in self._articles.values() if article.topic}

        return sorted(topics)

    def list_categories(self, *, min_articles: int = 1) -> list[str]:
        """
        List categories associated with enough distinct articles.

        Args:
            min_articles (int): Minimum number of articles required.

        Returns:
            list[str]: Sorted category names meeting the threshold.

        Raises:
            ValidationError: If min_articles is not positive.
        """

        self._check_open()

        if type(min_articles) is not int or min_articles < 1:
            raise ValidationError("min_articles must be positive")

        category_counts: dict[str, int] = {}

        for article in self._articles.values():
            # Count a category only once per article.
            for category in set(article.categories):
                category_counts[category] = category_counts.get(category, 0) + 1

        return sorted(
            category for category, count in category_counts.items() if count >= min_articles
        )

    def count_articles(self) -> int:
        """
        Count stored Wikipedia articles.

        Returns:
            int: Number of stored articles.
        """

        self._check_open()
        return len(self._articles)

    def count_chunks(self) -> int:
        """
        Count stored text chunks.

        Returns:
            int: Number of stored chunks.
        """

        self._check_open()
        return len(self._chunks)

    def set_ef_search(self, value: int) -> None:
        """
        Accept the HNSW search parameter for interface compatibility.

        The fake uses exact search, so this setting does not change
        its search algorithm.

        Args:
            value (int): Positive HNSW ef_search setting.

        Returns:
            None: Stores the setting without modifying search behavior.

        Raises:
            ValidationError: If value is not a positive integer.
        """

        self._check_open()

        if type(value) is not int or value < 1:
            raise ValidationError("ef_search must be positive")

        self._ef_search = value
