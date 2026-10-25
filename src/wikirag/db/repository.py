"""Provide the MariaDB-backed WikiRAG repository interface."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, Self

import pymysql

from wikirag.config import EMBEDDING_DIM, DbConfig, load_settings
from wikirag.db.connection import get_connection, transaction
from wikirag.db.queries import (
    COUNT_ARTICLES,
    COUNT_CHUNKS,
    COUNT_RESOLVED_LINKS,
    DELETE_ARTICLE_BY_ID,
    DELETE_ARTICLE_CATEGORIES,
    DELETE_ARTICLE_CHUNKS,
    DELETE_OUTGOING_LINKS,
    ENSURE_EMBEDDING_META,
    GET_ARTICLE_BY_ID,
    GET_ARTICLE_BY_TITLE,
    GET_ARTICLE_CATEGORIES,
    GET_ARTICLE_LINK_TITLES,
    GET_CATEGORY_ID,
    GET_CHUNK,
    GET_EMBEDDING_META,
    GET_FINGERPRINT,
    GET_FINGERPRINT_FOR_UPDATE,
    GET_LINKED_ARTICLES,
    GET_NEIGHBOR_CHUNKS,
    INSERT_ARTICLE,
    INSERT_ARTICLE_CATEGORY,
    INSERT_CHUNK,
    INSERT_OUTGOING_LINK,
    LIST_CATEGORIES,
    LIST_TOPICS,
    RESOLVE_LINK_TARGETS,
    SEARCH_CHUNKS_SEMANTIC,
    UPDATE_ARTICLE,
    UPSERT_CATEGORY,
    build_hybrid_search_query,
)
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
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


class WikiRepository:
    """Access WikiRAG data stored in MariaDB."""

    def __init__(self, config: DbConfig | None = None) -> None:
        """Open a database connection for repository operations.

        Args:
            config: Optional database configuration. Defaults to
                the application's configured MariaDB database.

        Raises:
            DatabaseConnectionError: If MariaDB cannot be reached.
        """
        self._config = config if config is not None else load_settings().db
        self._conn = get_connection(self._config)
        self._closed = False

    def _check_open(self) -> None:
        """Ensure the repository is still available.

        Raises:
            DatabaseConnectionError: If the repository was closed.
        """
        if self._closed:
            raise DatabaseConnectionError("Repository is closed")

    def __enter__(self) -> Self:
        """Enter the repository context manager.

        Returns:
            WikiRepository: The active repository.

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
        """Close the repository when leaving a context manager.

        Args:
            exc_type: Type of an active exception, if any.
            exc_value: Active exception, if any.
            traceback: Traceback of an active exception, if any.
        """
        self.close()

    def close(self) -> None:
        """Close the database connection.

        Calling close more than once has no additional effect.
        """
        if not self._closed:
            try:
                self._conn.close()
            finally:
                self._closed = True

    def _fetch_one(
        self,
        statement: str,
        params: tuple[object, ...] = (),
    ) -> dict[str, Any] | None:
        """Execute a read-only SQL query and retrieve one row.

        Args:
            statement: Parameterized SQL SELECT statement.
            params: Values for the query placeholders.

        Returns:
            dict[str, Any] | None: Result row or None if absent.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the database query fails.
        """
        self._check_open()

        try:
            with self._conn.cursor() as cursor:
                cursor.execute(statement, params)
                return cursor.fetchone()
        except pymysql.err.OperationalError:
            raise DatabaseConnectionError("Database connection failed during query") from None
        except pymysql.MySQLError:
            raise DatabaseError("Database query failed") from None

    def ensure_embedding_meta(self, model_name: str, dim: int) -> None:
        """Store or verify the embedding model configuration.

        Args:
            model_name: Name of the model generating embeddings.
            dim: Expected number of embedding dimensions.

        Raises:
            ValidationError: If the model name is invalid.
            SchemaMismatchError: If model or dimensions conflict.
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If the database operation fails.
        """
        self._check_open()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Model name must not be blank")

        model_name = model_name.strip()

        if len(model_name) > 200:
            raise ValidationError("Model name exceeds 200 characters")

        if type(dim) is not int or dim != EMBEDDING_DIM:
            raise SchemaMismatchError(f"Embedding dimension must be {EMBEDDING_DIM}")

        # The transaction keeps metadata insertion and validation together.
        with transaction(self._conn) as conn, conn.cursor() as cursor:
            cursor.execute(
                ENSURE_EMBEDDING_META,
                (model_name, dim),
            )
            cursor.execute(GET_EMBEDDING_META)
            stored = cursor.fetchone()

            if stored is None:
                raise SchemaMismatchError("Embedding metadata was not stored")

            if stored["model_name"] != model_name or stored["embedding_dim"] != dim:
                raise SchemaMismatchError("Embedding model or dimension does not match")

    def count_articles(self) -> int:
        """Count stored Wikipedia articles.

        Returns:
            int: Number of articles in the database.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        row = self._fetch_one(COUNT_ARTICLES)
        return int(row["total"]) if row is not None else 0

    def count_chunks(self) -> int:
        """Count stored Wikipedia text chunks.

        Returns:
            int: Number of chunks in the database.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        row = self._fetch_one(COUNT_CHUNKS)
        return int(row["total"]) if row is not None else 0

    def get_fingerprint(self, article_id: int) -> str | None:
        """Retrieve the stored fingerprint of a Wikipedia article.

        Args:
            article_id: Wikipedia page identifier.

        Returns:
            str | None: SHA-256 fingerprint, or None if not found.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        row = self._fetch_one(GET_FINGERPRINT, (article_id,))
        return str(row["fingerprint"]) if row is not None else None

    @staticmethod
    def _validate_article(article: ArticleRecord) -> None:
        """Validate article metadata before beginning any database writes.

        Args:
            article: Wikipedia article metadata to validate.

        Raises:
            ValidationError: If a field violates the database contract.
        """
        if not isinstance(article, ArticleRecord):
            raise ValidationError("Expected an ArticleRecord")

        if type(article.article_id) is not int or article.article_id <= 0:
            raise ValidationError("article_id must be a positive integer")

        if type(article.revision_id) is not int or article.revision_id < 0:
            raise ValidationError("revision_id must be a non-negative integer")

        if not isinstance(article.title, str) or not article.title.strip():
            raise ValidationError("Article title cannot be blank")

        if len(article.title) > 255:
            raise ValidationError("Article title exceeds 255 characters")

        if not isinstance(article.url, str) or not article.url.strip():
            raise ValidationError("Article URL cannot be blank")

        if len(article.url) > 512:
            raise ValidationError("Article URL exceeds 512 characters")

        if type(article.length_chars) is not int or article.length_chars < 0:
            raise ValidationError("Article length must be non-negative")

        if not isinstance(article.fingerprint, str) or (
            len(article.fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in article.fingerprint.lower())
        ):
            raise ValidationError("Fingerprint must be a SHA-256 hexadecimal digest")

        if not isinstance(article.language, str) or not 1 <= len(article.language) <= 2:
            raise ValidationError("Language code must contain 1 or 2 characters")

        if article.topic is not None and (
            not isinstance(article.topic, str) or len(article.topic) > 64
        ):
            raise ValidationError("Article topic exceeds database limits")

        for timestamp in (article.revised_at, article.fetched_at):
            if not isinstance(timestamp, datetime) or timestamp.utcoffset() is not None:
                raise ValidationError("Article timestamps must be naive UTC datetimes")

        for names, label in (
            (article.categories, "Category"),
            (article.linked_titles, "Linked title"),
        ):
            if not isinstance(names, tuple):
                raise ValidationError(f"{label} collection must be a tuple")

            for name in names:
                if not isinstance(name, str) or not name.strip() or len(name) > 255:
                    raise ValidationError(f"{label} must contain 1–255 characters")

    @staticmethod
    def _prepare_chunks(
        chunks: Sequence[ChunkRecord],
    ) -> list[tuple[ChunkRecord, str]]:
        """Validate chunks and serialize their embeddings for MariaDB.

        Args:
            chunks: Ordered text chunks with 384-dimensional embeddings.

        Returns:
            list[tuple[ChunkRecord, str]]: Valid chunks paired with
                their JSON-encoded vectors.

        Raises:
            ValidationError: If chunk ordering or text is invalid.
            EmbeddingDimensionError: If an embedding is invalid.
        """
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise ValidationError("Chunks must be an ordered sequence")

        if not chunks:
            raise ValidationError("An article must contain at least one chunk")

        prepared: list[tuple[ChunkRecord, str]] = []

        for expected_index, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkRecord):
                raise ValidationError("Expected a ChunkRecord")

            if type(chunk.chunk_index) is not int or chunk.chunk_index != expected_index:
                raise ValidationError("Chunk indices must start at zero and be contiguous")

            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ValidationError("Chunk text cannot be blank")

            # MariaDB TEXT stores at most about 64 KB of UTF-8 data.
            if len(chunk.text.encode("utf-8")) > 65000:
                raise ValidationError("Chunk text exceeds the database TEXT limit")

            if chunk.section_title is not None and (
                not isinstance(chunk.section_title, str) or len(chunk.section_title) > 255
            ):
                raise ValidationError("Chunk section title is invalid")

            try:
                vector = tuple(float(value) for value in chunk.embedding)
            except (TypeError, ValueError, OverflowError):
                raise EmbeddingDimensionError("Embedding must be numeric") from None

            if len(vector) != EMBEDDING_DIM:
                raise EmbeddingDimensionError(
                    f"Expected {EMBEDDING_DIM} dimensions, got {len(vector)}"
                )

            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingDimensionError("Embedding contains non-finite values")

            try:
                # Ensure every value can be represented as a 32-bit float.
                values32 = [struct.unpack("<f", struct.pack("<f", value))[0] for value in vector]
            except (OverflowError, struct.error):
                raise EmbeddingDimensionError(
                    "Embedding contains values outside float32 range"
                ) from None

            if not any(value != 0.0 for value in values32):
                raise EmbeddingDimensionError("Embedding cannot be a zero vector")

            prepared.append((chunk, json.dumps(values32, allow_nan=False)))

        return prepared

    def ingest_article(
        self,
        article: ArticleRecord,
        chunks: Sequence[ChunkRecord],
    ) -> IngestOutcome:
        """Insert, update, or skip a Wikipedia article atomically.

        Stores article metadata, text chunks, native vectors, category
        associations and outgoing links in one MariaDB transaction.

        Args:
            article: Article metadata, categories and outgoing links.
            chunks: Ordered text chunks and their embedding vectors.

        Returns:
            IngestOutcome: "inserted" for a new article, "updated"
                for a changed article, or "skipped" when its
                fingerprint is unchanged.

        Raises:
            ValidationError: If article or chunk data is invalid.
            EmbeddingDimensionError: If an embedding is invalid.
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If a database write fails.
        """
        self._check_open()

        # Validate every input before starting a database transaction.
        self._validate_article(article)
        prepared_chunks = self._prepare_chunks(chunks)

        with transaction(self._conn) as conn, conn.cursor() as cursor:
            cursor.execute(
                GET_FINGERPRINT_FOR_UPDATE,
                (article.article_id,),
            )
            previous = cursor.fetchone()

            if previous is not None and previous["fingerprint"] == article.fingerprint:
                return "skipped"

            if previous is None:
                outcome: IngestOutcome = "inserted"

                cursor.execute(
                    INSERT_ARTICLE,
                    (
                        article.article_id,
                        article.title,
                        article.url,
                        article.revision_id,
                        article.revised_at,
                        article.length_chars,
                        article.fingerprint,
                        article.topic,
                        article.language,
                        article.fetched_at,
                    ),
                )
            else:
                outcome = "updated"

                cursor.execute(
                    UPDATE_ARTICLE,
                    (
                        article.title,
                        article.url,
                        article.revision_id,
                        article.revised_at,
                        article.length_chars,
                        article.fingerprint,
                        article.topic,
                        article.language,
                        article.fetched_at,
                        article.article_id,
                    ),
                )

                # Replace dependent rows only for changed articles.
                cursor.execute(DELETE_ARTICLE_CHUNKS, (article.article_id,))
                cursor.execute(DELETE_ARTICLE_CATEGORIES, (article.article_id,))
                cursor.execute(DELETE_OUTGOING_LINKS, (article.article_id,))

            for chunk, vector_json in prepared_chunks:
                cursor.execute(
                    INSERT_CHUNK,
                    (
                        article.article_id,
                        chunk.chunk_index,
                        chunk.section_title,
                        chunk.text,
                        len(chunk.text),
                        vector_json,
                    ),
                )

            for category in dict.fromkeys(article.categories):
                cursor.execute(UPSERT_CATEGORY, (category,))
                cursor.execute(GET_CATEGORY_ID, (category,))
                category_row = cursor.fetchone()

                if category_row is None:
                    raise DatabaseError("Category lookup failed after insertion")

                cursor.execute(
                    INSERT_ARTICLE_CATEGORY,
                    (article.article_id, category_row["category_id"]),
                )

            for target_title in dict.fromkeys(article.linked_titles):
                # Resolve target IDs later, after all articles are ingested.
                cursor.execute(
                    INSERT_OUTGOING_LINK,
                    (article.article_id, target_title),
                )

            return outcome

    def _fetch_all(
        self,
        statement: str,
        params: tuple[object, ...] = (),
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return all matching rows.

        Args:
            statement: Parameterized SQL SELECT statement.
            params: Query parameter values.

        Returns:
            list[dict[str, Any]]: Rows returned by MariaDB.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        self._check_open()

        try:
            with self._conn.cursor() as cursor:
                cursor.execute(statement, params)
                return list(cursor.fetchall())
        except pymysql.err.OperationalError:
            raise DatabaseConnectionError("Database connection failed during query") from None
        except pymysql.MySQLError:
            raise DatabaseError("Database query failed") from None

    def _article_from_row(
        self,
        row: dict[str, Any],
    ) -> ArticleRecord:
        """Convert a database row into a complete ArticleRecord.

        Args:
            row: Article metadata retrieved from MariaDB.

        Returns:
            ArticleRecord: Metadata, categories, and outgoing link titles.

        Raises:
            DatabaseConnectionError: If the connection fails.
            DatabaseError: If related metadata cannot be retrieved.
        """
        article_id = int(row["article_id"])

        category_rows = self._fetch_all(GET_ARTICLE_CATEGORIES, (article_id,))
        link_rows = self._fetch_all(GET_ARTICLE_LINK_TITLES, (article_id,))

        return ArticleRecord(
            article_id=article_id,
            title=row["title"],
            url=row["url"],
            revision_id=int(row["revision_id"]),
            fingerprint=row["fingerprint"],
            revised_at=row["revised_at"],
            length_chars=int(row["length_chars"]),
            fetched_at=row["fetched_at"],
            topic=row["topic"],
            language=row["language"],
            categories=tuple(item["name"] for item in category_rows),
            linked_titles=tuple(item["target_title"] for item in link_rows),
        )

    def get_article(self, article_id: int) -> ArticleRecord | None:
        """Retrieve a Wikipedia article by its page identifier.

        Args:
            article_id: Wikipedia page identifier.

        Returns:
            ArticleRecord | None: Full article metadata or None.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If a database query fails.
        """
        row = self._fetch_one(GET_ARTICLE_BY_ID, (article_id,))

        if row is None:
            return None

        return self._article_from_row(row)

    def get_article_by_title(self, title: str) -> ArticleRecord | None:
        """Retrieve an article using exact, case-sensitive title matching.

        Args:
            title: Wikipedia article title.

        Returns:
            ArticleRecord | None: Matching article or None.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If a database query fails.
        """
        row = self._fetch_one(GET_ARTICLE_BY_TITLE, (title,))

        if row is None:
            return None

        return self._article_from_row(row)

    @staticmethod
    def _chunk_from_row(row: dict[str, Any]) -> StoredChunk:
        """Convert a MariaDB result row into a StoredChunk.

        Args:
            row: Database row containing chunk and article information.

        Returns:
            StoredChunk: Structured chunk with its source article title.
        """
        return StoredChunk(
            chunk_id=int(row["chunk_id"]),
            article_id=int(row["article_id"]),
            article_title=row["article_title"],
            chunk_index=int(row["chunk_index"]),
            section_title=row["section_title"],
            text=row["text"],
        )

    def get_chunk(self, chunk_id: int) -> StoredChunk | None:
        """Retrieve a stored chunk by its database identifier.

        Args:
            chunk_id: Unique identifier of the chunk.

        Returns:
            StoredChunk | None: Matching chunk or None.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        row = self._fetch_one(GET_CHUNK, (chunk_id,))

        if row is None:
            return None

        return self._chunk_from_row(row)

    def get_neighbor_chunks(
        self,
        chunk_id: int,
        *,
        window: int = 1,
    ) -> list[StoredChunk]:
        """Retrieve a chunk and its neighbors within the same article.

        Args:
            chunk_id: Identifier of the central chunk.
            window: Number of chunk positions to include on each side.

        Returns:
            list[StoredChunk]: Chunks in ascending chunk-index order,
                or an empty list if the central chunk is missing.

        Raises:
            ValidationError: If window is negative or not an integer.
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If a query fails.
        """
        self._check_open()

        if type(window) is not int or window < 0:
            raise ValidationError("window must be a non-negative integer")

        center = self.get_chunk(chunk_id)

        if center is None:
            return []

        rows = self._fetch_all(
            GET_NEIGHBOR_CHUNKS,
            (
                center.article_id,
                max(0, center.chunk_index - window),
                center.chunk_index + window,
            ),
        )

        return [self._chunk_from_row(row) for row in rows]

    def list_topics(self) -> list[str]:
        """List distinct, non-empty Wikipedia corpus topics.

        Returns:
            list[str]: Topic names in alphabetical order.

        Raises:
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        rows = self._fetch_all(LIST_TOPICS)
        return [row["topic"] for row in rows]

    def list_categories(self, *, min_articles: int = 1) -> list[str]:
        """List categories associated with enough stored articles.

        Args:
            min_articles: Minimum number of distinct articles required.

        Returns:
            list[str]: Matching category names in alphabetical order.

        Raises:
            ValidationError: If min_articles is not positive.
            DatabaseConnectionError: If the connection is unavailable.
            DatabaseError: If the query fails.
        """
        self._check_open()

        if type(min_articles) is not int or min_articles < 1:
            raise ValidationError("min_articles must be a positive integer")

        rows = self._fetch_all(LIST_CATEGORIES, (min_articles,))
        return [row["name"] for row in rows]

    def resolve_links(self) -> int:
        """Resolve Wikipedia link titles to stored article identifiers.

        Links to articles outside the corpus remain unresolved.
        Running this method repeatedly is safe.

        Returns:
            int: Total number of resolved links.

        Raises:
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If a database operation fails.
        """
        self._check_open()

        with transaction(self._conn) as conn, conn.cursor() as cursor:
            # Recalculate targets to handle changed article titles.
            cursor.execute(RESOLVE_LINK_TARGETS)
            cursor.execute(COUNT_RESOLVED_LINKS)
            row = cursor.fetchone()

            return int(row["total"])

    def get_linked_articles(self, article_id: int) -> list[ArticleRecord]:
        """Retrieve articles linked from a source Wikipedia article.

        Only links resolved to articles in our corpus are returned.

        Args:
            article_id: Wikipedia identifier of the source article.

        Returns:
            list[ArticleRecord]: Linked articles ordered by article ID.

        Raises:
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If a database query fails.
        """
        rows = self._fetch_all(GET_LINKED_ARTICLES, (article_id,))

        return [self._article_from_row(row) for row in rows]

    def delete_article(self, article_id: int) -> bool:
        """Delete an article and its dependent database records.

        The database's foreign keys remove outgoing links,
        category associations and chunks. Incoming links retain
        their titles but lose their resolved target ID.

        Args:
            article_id: Wikipedia identifier of the article to delete.

        Returns:
            bool: True if the article existed, otherwise False.

        Raises:
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If deletion fails.
        """
        self._check_open()

        with transaction(self._conn) as conn, conn.cursor() as cursor:
            cursor.execute(DELETE_ARTICLE_BY_ID, (article_id,))
            return cursor.rowcount > 0

    @staticmethod
    def _serialize_search_vector(query_embedding: Sequence[float]) -> str:
        """Validate a query embedding and serialize it for MariaDB.

        The serialized JSON vector is passed to VEC_FromText in SQL.

        Args:
            query_embedding: Numeric query embedding containing 384 values.

        Returns:
            str: JSON array containing finite float32-compatible values.

        Raises:
            EmbeddingDimensionError: If the query has incorrect dimensions,
                non-numeric values, non-finite values, or zero magnitude.
        """
        if isinstance(query_embedding, (str, bytes)):
            raise EmbeddingDimensionError("Query embedding must be numeric")

        try:
            values = tuple(float(value) for value in query_embedding)
        except (TypeError, ValueError, OverflowError):
            raise EmbeddingDimensionError("Query embedding must contain numbers") from None

        if len(values) != EMBEDDING_DIM:
            raise EmbeddingDimensionError(f"Expected {EMBEDDING_DIM} dimensions, got {len(values)}")

        if not all(math.isfinite(value) for value in values):
            raise EmbeddingDimensionError("Query embedding contains non-finite values")

        try:
            # MariaDB VECTOR values are stored as 32-bit floating-point numbers.
            values32 = [struct.unpack("<f", struct.pack("<f", value))[0] for value in values]
        except (OverflowError, struct.error):
            raise EmbeddingDimensionError(
                "Query embedding contains values outside float32 range"
            ) from None

        if not any(value != 0.0 for value in values32):
            raise EmbeddingDimensionError("Query embedding cannot be a zero vector")

        return json.dumps(values32, allow_nan=False)

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Retrieve ranked chunks using MariaDB native cosine search.

        Unfiltered searches use the HNSW-eligible semantic query.
        Filtered searches apply relational SQL constraints before
        exact cosine ranking to avoid selective-filter under-fill.

        Args:
            query_embedding: Numeric query embedding containing 384 values.
            top_k: Maximum number of results, between 1 and 100.
            filters: Optional relational and distance constraints.

        Returns:
            list[SearchHit]: Ranked chunks with article metadata,
                cosine distance, and similarity score.

        Raises:
            ValidationError: If search parameters are invalid.
            EmbeddingDimensionError: If the query embedding is invalid.
            DatabaseConnectionError: If MariaDB is unavailable.
            DatabaseError: If the SQL query fails.
        """

        self._check_open()

        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValidationError("top_k must be an integer between 1 and 100")

        if filters is not None and not isinstance(filters, SearchFilters):
            raise ValidationError("filters must be SearchFilters or None")

        if filters is not None and filters.max_distance is not None:
            threshold = filters.max_distance

            if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or not 0.0 <= threshold <= 2.0
            ):
                raise ValidationError("max_distance must be between 0 and 2")

        vector_json = self._serialize_search_vector(query_embedding)

        if filters is not None and not filters.is_empty():
            # Filter eligible articles before ranking to avoid top-K under-fill.
            statement, params = build_hybrid_search_query(
                filters,
                vector_json=vector_json,
                top_k=top_k,
            )
        else:
            # Preserve the HNSW-eligible query for unfiltered searches.
            statement = SEARCH_CHUNKS_SEMANTIC
            params = (vector_json, vector_json, top_k)

        rows = self._fetch_all(statement, params)

        hits: list[SearchHit] = []

        for row in rows:
            distance = float(row["distance"])

            hits.append(
                SearchHit(
                    chunk_id=int(row["chunk_id"]),
                    article_id=int(row["article_id"]),
                    article_title=row["article_title"],
                    article_url=row["article_url"],
                    chunk_index=int(row["chunk_index"]),
                    section_title=row["section_title"],
                    text=row["text"],
                    distance=distance,
                    score=1.0 - distance,
                )
            )

        return hits

    def search_chunks_exact(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Search chunks using an exact cosine-distance baseline.

        This method is reserved for evaluation against HNSW results.

        Args:
            query_embedding: Query vector with 384 dimensions.
            top_k: Maximum number of matching chunks to return.
            filters: Optional relational search constraints.

        Returns:
            list[SearchHit]: Exact ranked results when implemented.

        Raises:
            NotImplementedError: Exact vector search is not yet implemented.
        """
        raise NotImplementedError(
            "search_chunks_exact will be implemented in feature/database-search"
        )

    def set_ef_search(self, value: int) -> None:
        """Set the HNSW search parameter for the current MariaDB session.

        This method is reserved for the database-search implementation.

        Args:
            value: Positive integer controlling HNSW search breadth.

        Returns:
            None: The session setting is updated when implemented.

        Raises:
            NotImplementedError: HNSW search configuration is not yet implemented.
        """
        raise NotImplementedError("set_ef_search will be implemented in feature/database-search")
