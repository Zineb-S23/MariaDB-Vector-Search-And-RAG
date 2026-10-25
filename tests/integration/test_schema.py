"""Integration tests for the real MariaDB WikiRAG schema."""

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import patch

import pytest
from pymysql.connections import Connection

from wikirag.config import load_settings
from wikirag.db.connection import get_connection
from wikirag.db.schema import init_schema, verify_schema
from wikirag.errors import SchemaMismatchError, ValidationError


@pytest.fixture
def test_db_conn() -> Iterator[Connection]:
    """Provide an isolated, initialized MariaDB test database.

    Yields:
        Connection: Connection to the configured test database.

    Raises:
        ValidationError: If the test database name is unsafe.
    """
    settings = load_settings()

    if not settings.db_name_test.endswith("_test") or settings.db_name_test == settings.db.database:
        raise ValidationError("Tests require a separate *_test database")

    # Change only the database name; reuse the application credentials.
    config = replace(settings.db, database=settings.db_name_test)
    conn = get_connection(config)

    try:
        # Rebuild only the test database before each test.
        init_schema(conn, recreate=True)
        yield conn
    finally:
        conn.close()


def test_recreate_schema_creates_six_tables(
    test_db_conn: Connection,
) -> None:
    """Confirm schema recreation produces all required MariaDB tables.

    Args:
        test_db_conn: Connection to the isolated test database.
    """
    verify_schema(test_db_conn)

    with test_db_conn.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = {next(iter(row.values())) for row in cursor.fetchall()}

    assert tables == {
        "embedding_meta",
        "articles",
        "chunks",
        "categories",
        "article_categories",
        "links",
    }

    with test_db_conn.cursor() as cursor:
        cursor.execute("SHOW CREATE TABLE chunks")
        definition = cursor.fetchone()["Create Table"]

    assert "vector(384)" in definition.lower()
    assert "VECTOR KEY" in definition
    assert "cosine" in definition


def test_init_schema_creates_tables_from_empty_database(
    test_db_conn: Connection,
) -> None:
    """Create all tables from an empty test database.

    Args:
        test_db_conn: Connection to the isolated test database.
    """
    # Drop child tables before parent tables to satisfy foreign keys.
    tables = (
        "links",
        "article_categories",
        "chunks",
        "categories",
        "articles",
        "embedding_meta",
    )

    with test_db_conn.cursor() as cursor:
        for table in tables:
            cursor.execute(f"DROP TABLE `{table}`")

    init_schema(test_db_conn)
    verify_schema(test_db_conn)


def test_init_schema_preserves_existing_data(
    test_db_conn: Connection,
) -> None:
    """Verify repeated initialization does not erase existing data.

    Args:
        test_db_conn: Connection to the isolated test database.
    """
    with test_db_conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO embedding_meta (id, model_name, embedding_dim) VALUES (%s, %s, %s)",
            (1, "integration-test-model", 384),
        )

    test_db_conn.commit()

    # Existing valid tables should be verified rather than recreated.
    init_schema(test_db_conn)
    init_schema(test_db_conn)

    with test_db_conn.cursor() as cursor:
        cursor.execute(
            "SELECT model_name FROM embedding_meta WHERE id = %s",
            (1,),
        )
        row = cursor.fetchone()

    assert row["model_name"] == "integration-test-model"


def test_partial_schema_is_rejected(
    test_db_conn: Connection,
) -> None:
    """Reject an incomplete schema instead of silently rebuilding it.

    Args:
        test_db_conn: Connection to the isolated test database.
    """
    with test_db_conn.cursor() as cursor:
        cursor.execute("DROP TABLE links")

    with pytest.raises(SchemaMismatchError):
        verify_schema(test_db_conn)

    with pytest.raises(SchemaMismatchError):
        init_schema(test_db_conn)


def test_recreation_requires_configured_test_database(
    test_db_conn: Connection,
) -> None:
    """Prevent destructive recreation of an unauthorized database.

    Args:
        test_db_conn: Connection to the isolated test database.
    """
    settings = load_settings()
    alternate_settings = replace(
        settings,
        db_name_test="different_test",
    )

    # Simulate a configuration that does not authorize this database.
    with (
        patch(
            "wikirag.db.schema.load_settings",
            return_value=alternate_settings,
        ),
        pytest.raises(ValidationError, match="DB_NAME_TEST"),
    ):
        init_schema(test_db_conn, recreate=True)

    # The rejected operation must leave the real tables intact.
    verify_schema(test_db_conn)
