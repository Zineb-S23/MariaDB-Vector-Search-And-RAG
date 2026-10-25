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
    DELETE_ARTICLE_CATEGORIES,
    DELETE_ARTICLE_CHUNKS,
    DELETE_OUTGOING_LINKS,
    ENSURE_EMBEDDING_META,
    GET_CATEGORY_ID,
    GET_EMBEDDING_META,
    GET_FINGERPRINT,
    GET_FINGERPRINT_FOR_UPDATE,
    INSERT_ARTICLE,
    INSERT_ARTICLE_CATEGORY,
    INSERT_CHUNK,
    INSERT_OUTGOING_LINK,
    UPDATE_ARTICLE,
    UPSERT_CATEGORY,
)
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    EmbeddingDimensionError,
    SchemaMismatchError,
    ValidationError,
)
from wikirag.models import ArticleRecord, ChunkRecord, IngestOutcome


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
