"""Unit tests for MariaDB connections and transaction handling."""

from unittest.mock import MagicMock, patch

import pymysql
import pytest

from wikirag.config import DbConfig
from wikirag.db.connection import (
    check_connection,
    get_connection,
    transaction,
)
from wikirag.errors import DatabaseConnectionError, DatabaseError


@pytest.fixture
def db_config() -> DbConfig:
    """Provide safe, non-production database settings for unit tests.

    Returns:
        DbConfig: A sample configuration with placeholder credentials.
    """
    return DbConfig(
        host="127.0.0.1",
        port=13307,
        database="wikirag_test",
        user="test_user",
        password="placeholder_password",
    )


def test_get_connection_uses_configuration(db_config: DbConfig) -> None:
    """Verify that PyMySQL receives the expected connection settings."""
    with patch("wikirag.db.connection.pymysql.connect") as mock_connect:
        connection = get_connection(db_config)

    assert connection is mock_connect.return_value

    mock_connect.assert_called_once_with(
        host=db_config.host,
        port=db_config.port,
        database=db_config.database,
        user=db_config.user,
        password=db_config.password,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
        cursorclass=pymysql.cursors.DictCursor,
    )


def test_get_connection_handles_failure(db_config: DbConfig) -> None:
    """Convert PyMySQL connection failures into project-level errors."""
    with (
        patch(
            "wikirag.db.connection.pymysql.connect",
            side_effect=pymysql.err.OperationalError(
                1045, "Access denied for placeholder_password"
            ),
        ),
        pytest.raises(DatabaseConnectionError) as exc_info,
    ):
        get_connection(db_config)

    # Database credentials must not appear in public exception messages.
    assert "placeholder_password" not in str(exc_info.value)


def test_check_connection_success(db_config: DbConfig) -> None:
    """Return True when MariaDB responds successfully to SELECT 1."""
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
    mock_cursor.fetchone.return_value = {"1": 1}

    with patch(
        "wikirag.db.connection.get_connection",
        return_value=mock_conn,
    ):
        result = check_connection(db_config)

    assert result is True
    mock_cursor.execute.assert_called_once_with("SELECT 1")
    mock_conn.close.assert_called_once()


def test_check_connection_failure(db_config: DbConfig) -> None:
    """Return False when a database connection cannot be established."""
    with patch(
        "wikirag.db.connection.get_connection",
        side_effect=DatabaseConnectionError("Connection failed"),
    ):
        assert check_connection(db_config) is False


def test_check_connection_query_failure(db_config: DbConfig) -> None:
    """Return False and close the connection when SELECT 1 fails."""
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
    mock_cursor.execute.side_effect = pymysql.MySQLError("Query failed")

    with patch(
        "wikirag.db.connection.get_connection",
        return_value=mock_conn,
    ):
        result = check_connection(db_config)

    assert result is False
    mock_conn.close.assert_called_once()


def test_check_connection_empty_result(db_config: DbConfig) -> None:
    """Return False if the health-check query produces no row."""
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
    mock_cursor.fetchone.return_value = None

    with patch(
        "wikirag.db.connection.get_connection",
        return_value=mock_conn,
    ):
        result = check_connection(db_config)

    assert result is False
    mock_conn.close.assert_called_once()


def test_transaction_commits_on_success() -> None:
    """Commit when all operations inside the transaction succeed."""
    mock_conn = MagicMock()

    with transaction(mock_conn) as conn:
        assert conn is mock_conn

    mock_conn.begin.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()


def test_transaction_rolls_back_on_python_error() -> None:
    """Roll back and preserve a non-database exception."""
    mock_conn = MagicMock()

    with pytest.raises(ValueError, match="Invalid data"), transaction(mock_conn):
        raise ValueError("Invalid data")

    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()


def test_transaction_handles_sql_error() -> None:
    """Roll back and translate a SQL failure into DatabaseError."""
    mock_conn = MagicMock()

    with pytest.raises(DatabaseError, match="MariaDB transaction failed"), transaction(mock_conn):
        raise pymysql.err.IntegrityError(1062, "Duplicate entry")

    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()


def test_transaction_handles_connection_loss() -> None:
    """Roll back and identify operational connection failures."""
    mock_conn = MagicMock()

    with pytest.raises(DatabaseConnectionError), transaction(mock_conn):
        raise pymysql.err.OperationalError(2006, "MySQL server has gone away")

    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()


def test_transaction_handles_commit_failure() -> None:
    """Attempt rollback when the transaction commit fails."""
    mock_conn = MagicMock()
    mock_conn.commit.side_effect = pymysql.MySQLError("Commit failed")

    with pytest.raises(DatabaseError, match="MariaDB transaction failed"), transaction(mock_conn):
        pass

    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_called_once()


def test_transaction_handles_rollback_failure() -> None:
    """Report a DatabaseError if rolling back also fails."""
    mock_conn = MagicMock()
    mock_conn.rollback.side_effect = pymysql.MySQLError("Rollback failed")

    with pytest.raises(DatabaseError, match="rollback failed"), transaction(mock_conn):
        raise ValueError("Invalid data")

    mock_conn.rollback.assert_called_once()
