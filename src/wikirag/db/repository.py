"""Provide the MariaDB-backed WikiRAG repository interface."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import pymysql

from wikirag.config import EMBEDDING_DIM, DbConfig, load_settings
from wikirag.db.connection import get_connection, transaction
from wikirag.db.queries import (
    COUNT_ARTICLES,
    COUNT_CHUNKS,
    ENSURE_EMBEDDING_META,
    GET_EMBEDDING_META,
    GET_FINGERPRINT,
)
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    SchemaMismatchError,
    ValidationError,
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
