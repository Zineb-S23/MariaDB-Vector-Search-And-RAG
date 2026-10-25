"""Integration tests for the MariaDB-backed WikiRepository."""

from contextlib import closing
from dataclasses import replace
from datetime import datetime

import pytest

from wikirag.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    DbConfig,
    load_settings,
)
from wikirag.db.connection import get_connection
from wikirag.db.repository import WikiRepository
from wikirag.db.schema import init_schema
from wikirag.errors import (
    DatabaseConnectionError,
    SchemaMismatchError,
    ValidationError,
)


@pytest.fixture
def test_db_config() -> DbConfig:
    """Initialize an isolated MariaDB database for one test.

    Returns:
        DbConfig: Connection settings for the test database.

    Raises:
        ValidationError: If test database configuration is unsafe.
    """
    settings = load_settings()

    if not settings.db_name_test.endswith("_test") or settings.db_name_test == settings.db.database:
        raise ValidationError("Integration tests require a separate test database")

    config = replace(settings.db, database=settings.db_name_test)

    # Only the dedicated test database is rebuilt.
    with closing(get_connection(config)) as conn:
        init_schema(conn, recreate=True)

    return config


def test_repository_starts_empty(test_db_config: DbConfig) -> None:
    """Verify repository counts and missing fingerprints.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 0
        assert repo.count_chunks() == 0
        assert repo.get_fingerprint(123456) is None


def test_embedding_metadata_is_idempotent(
    test_db_config: DbConfig,
) -> None:
    """Store embedding metadata without duplicating it.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

        # Calling again with identical values must be safe.
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT model_name, embedding_dim FROM embedding_meta")
        rows = cursor.fetchall()

    assert len(rows) == 1
    assert rows[0]["model_name"] == EMBEDDING_MODEL_NAME
    assert rows[0]["embedding_dim"] == EMBEDDING_DIM


def test_embedding_metadata_rejects_conflicts(
    test_db_config: DbConfig,
) -> None:
    """Reject incompatible embedding models without overwriting metadata.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

        with pytest.raises(SchemaMismatchError):
            repo.ensure_embedding_meta(
                "different-embedding-model",
                EMBEDDING_DIM,
            )

        with pytest.raises(SchemaMismatchError):
            repo.ensure_embedding_meta(
                EMBEDDING_MODEL_NAME,
                768,
            )

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT model_name, embedding_dim FROM embedding_meta WHERE id = 1")
        row = cursor.fetchone()

    # Rejected calls must leave the original settings unchanged.
    assert row["model_name"] == EMBEDDING_MODEL_NAME
    assert row["embedding_dim"] == EMBEDDING_DIM


def test_embedding_metadata_rejects_blank_name(
    test_db_config: DbConfig,
) -> None:
    """Reject invalid model names before writing to MariaDB.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        with pytest.raises(ValidationError):
            repo.ensure_embedding_meta("   ", EMBEDDING_DIM)

        assert repo.count_articles() == 0

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS total FROM embedding_meta")
        row = cursor.fetchone()

    assert row["total"] == 0


def test_repository_reads_stored_article(
    test_db_config: DbConfig,
) -> None:
    """Count a real article and retrieve its SHA-256 fingerprint.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article_id = 987654
    fingerprint = "a" * 64
    timestamp = datetime(2025, 1, 1)

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO articles (
                article_id, title, url, revision_id,
                revised_at, length_chars, fingerprint,
                topic, language, fetched_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                article_id,
                "Integration Test Article",
                "https://en.wikipedia.org/wiki/Integration_Test_Article",
                123,
                timestamp,
                100,
                fingerprint,
                "testing",
                "en",
                timestamp,
            ),
        )
        conn.commit()

    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 1
        assert repo.count_chunks() == 0
        assert repo.get_fingerprint(article_id) == fingerprint
        assert repo.get_fingerprint(123456) is None


def test_repository_rejects_calls_after_close(
    test_db_config: DbConfig,
) -> None:
    """Prevent database operations after the repository is closed.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 0

    # Closing repeatedly should be harmless.
    repo.close()

    with pytest.raises(DatabaseConnectionError):
        repo.count_articles()

    with pytest.raises(DatabaseConnectionError):
        repo.get_fingerprint(123456)
