"""Create and verify the canonical MariaDB WikiRAG schema."""

from __future__ import annotations

import re
from pathlib import Path

import pymysql
from pymysql.connections import Connection

from wikirag.config import load_settings
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    SchemaMismatchError,
    ValidationError,
)

# Resolve the canonical SQL file from the repository root.
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "sql" / "001_schema.sql"

EXPECTED_COLUMNS = {
    "embedding_meta": {
        "id",
        "model_name",
        "embedding_dim",
        "created_at",
    },
    "articles": {
        "article_id",
        "title",
        "url",
        "revision_id",
        "revised_at",
        "length_chars",
        "fingerprint",
        "topic",
        "language",
        "fetched_at",
        "ingested_at",
    },
    "chunks": {
        "chunk_id",
        "article_id",
        "chunk_index",
        "section_title",
        "text",
        "char_count",
        "embedding",
    },
    "categories": {"category_id", "name"},
    "article_categories": {"article_id", "category_id"},
    "links": {
        "source_article_id",
        "target_title",
        "target_article_id",
    },
}

# Fixed statements avoid constructing SQL from user-supplied identifiers.
DROP_STATEMENTS = (
    "DROP TABLE IF EXISTS links",
    "DROP TABLE IF EXISTS article_categories",
    "DROP TABLE IF EXISTS chunks",
    "DROP TABLE IF EXISTS categories",
    "DROP TABLE IF EXISTS articles",
    "DROP TABLE IF EXISTS embedding_meta",
)

EXPECTED_FOREIGN_KEYS = {
    ("chunks", "article_id", "articles", "article_id"),
    ("article_categories", "article_id", "articles", "article_id"),
    ("article_categories", "category_id", "categories", "category_id"),
    ("links", "source_article_id", "articles", "article_id"),
    ("links", "target_article_id", "articles", "article_id"),
}


def _schema_statements() -> list[str]:
    """Read CREATE TABLE statements from the canonical SQL file.

    Returns:
        list[str]: Six schema creation statements in dependency order.

    Raises:
        SchemaMismatchError: If the SQL file is missing or malformed.
    """
    try:
        source = SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError:
        raise SchemaMismatchError("Cannot read sql/001_schema.sql") from None

    # Our canonical DDL uses full-line comments and simple statements.
    lines = [line for line in source.splitlines() if not line.lstrip().startswith("--")]

    statements = [part.strip() for part in "\n".join(lines).split(";") if part.strip()]

    if len(statements) != len(EXPECTED_COLUMNS) or any(
        not statement.upper().startswith("CREATE TABLE ") for statement in statements
    ):
        raise SchemaMismatchError("Expected exactly six CREATE TABLE statements")

    return statements


def _database_name(conn: Connection) -> str:
    """Get the currently selected MariaDB database.

    Args:
        conn: An existing MariaDB connection.

    Returns:
        str: Name of the selected database.

    Raises:
        ValidationError: If no database is selected.
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT DATABASE() AS db_name")
        row = cursor.fetchone()

    name = row["db_name"] if row else None

    if not name:
        raise ValidationError("No MariaDB database is selected")

    return name


def _existing_tables(conn: Connection) -> set[str]:
    """Retrieve table names from the selected database.

    Args:
        conn: An existing MariaDB connection.

    Returns:
        set[str]: Names of existing base tables.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_TYPE = 'BASE TABLE'"
        )
        return {row["TABLE_NAME"] for row in cursor.fetchall()}


def verify_schema(conn: Connection) -> None:
    """Verify the WikiRAG tables, columns, relationships and vector index.

    Args:
        conn: Open MariaDB connection using DictCursor.

    Returns:
        None: The schema matches the required database structure.

    Raises:
        SchemaMismatchError: If tables, columns, foreign keys or
            vector configuration do not match.
        DatabaseConnectionError: If the database connection fails.
        DatabaseError: If a metadata query fails.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE()"
            )

            columns: dict[str, dict[str, str]] = {}

            for row in cursor.fetchall():
                table = row["TABLE_NAME"]
                columns.setdefault(table, {})[row["COLUMN_NAME"]] = row["COLUMN_TYPE"]

            for table, required in EXPECTED_COLUMNS.items():
                actual = set(columns.get(table, {}))

                if actual != required:
                    raise SchemaMismatchError(
                        f"Schema mismatch in {table}: "
                        f"missing={sorted(required - actual)}, "
                        f"extra={sorted(actual - required)}"
                    )

            if columns["chunks"]["embedding"].lower() != "vector(384)":
                raise SchemaMismatchError("chunks.embedding must have type VECTOR(384)")

            cursor.execute("SHOW CREATE TABLE chunks")
            definition = cursor.fetchone()["Create Table"]

            # Normalize MariaDB's quoted SHOW CREATE TABLE output.
            normalized = re.sub(r"[\s`']", "", definition).lower()

            if (
                "vectorkeyembedding(embedding)" not in normalized
                or "m=16" not in normalized
                or "distance=cosine" not in normalized
            ):
                raise SchemaMismatchError("Expected a cosine VECTOR INDEX with M=16")

            cursor.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, "
                "REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
                "FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND REFERENCED_TABLE_NAME IS NOT NULL"
            )

            actual_keys = {
                (
                    row["TABLE_NAME"],
                    row["COLUMN_NAME"],
                    row["REFERENCED_TABLE_NAME"],
                    row["REFERENCED_COLUMN_NAME"],
                )
                for row in cursor.fetchall()
            }

            if not EXPECTED_FOREIGN_KEYS.issubset(actual_keys):
                raise SchemaMismatchError("Required database foreign keys are missing")

    except pymysql.err.OperationalError:
        raise DatabaseConnectionError("Connection failed while verifying schema") from None
    except pymysql.MySQLError:
        raise DatabaseError("Failed to inspect MariaDB schema") from None


def init_schema(conn: Connection, *, recreate: bool = False) -> None:
    """Create the schema or verify an existing WikiRAG installation.

    Destructive recreation is restricted to the configured test database.
    MariaDB DDL is not rolled back as one transaction.

    Args:
        conn: An existing MariaDB connection using DictCursor.
        recreate: Whether to rebuild the configured test database schema.

    Returns:
        None: The canonical schema exists and passes verification.

    Raises:
        ValidationError: If recreation targets a non-test database.
        SchemaMismatchError: If existing tables are incompatible.
        DatabaseConnectionError: If MariaDB loses its connection.
        DatabaseError: If schema creation or removal fails.
    """
    statements = _schema_statements()

    try:
        database = _database_name(conn)
        existing = _existing_tables(conn)
        required = set(EXPECTED_COLUMNS)

        if recreate:
            settings = load_settings()

            # Never allow automatic destruction of the application database.
            if database != settings.db_name_test or not database.endswith("_test"):
                raise ValidationError("recreate=True is allowed only for DB_NAME_TEST")

            if existing - required:
                raise SchemaMismatchError("Test database contains unrelated tables")

            with conn.cursor() as cursor:
                for statement in DROP_STATEMENTS:
                    cursor.execute(statement)

        elif existing:
            # An existing installation must match; do not overwrite it.
            verify_schema(conn)
            return

        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

        verify_schema(conn)

    except pymysql.err.OperationalError:
        raise DatabaseConnectionError("Connection failed while initializing schema") from None
    except pymysql.MySQLError:
        raise DatabaseError("Schema initialization failed; partial tables may exist") from None
