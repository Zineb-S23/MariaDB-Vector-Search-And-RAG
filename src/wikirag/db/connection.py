"""Manage MariaDB connections and database transactions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager

import pymysql
from pymysql.connections import Connection

from wikirag.config import DbConfig
from wikirag.errors import DatabaseConnectionError, DatabaseError


def get_connection(config: DbConfig) -> Connection:
    """Open a MariaDB connection using validated configuration.

    Args:
        config: Database host, port, name, user, and password.

    Returns:
        An open PyMySQL connection with transactions enabled.

    Raises:
        DatabaseConnectionError: If MariaDB cannot be reached or
            authentication fails.
    """
    try:
        return pymysql.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            charset="utf8mb4",
            autocommit=False,
            connect_timeout=10,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except pymysql.MySQLError:
        # Avoid exposing connection credentials in error messages.
        raise DatabaseConnectionError("Unable to connect to MariaDB") from None


def check_connection(config: DbConfig) -> bool:
    """Check whether MariaDB accepts connections and SQL queries.

    Args:
        config: Configuration of the database to check.

    Returns:
        True if a SELECT 1 query succeeds, otherwise False.
    """
    try:
        # Always close the connection, including when the query fails.
        with closing(get_connection(config)) as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() is not None
    except (DatabaseConnectionError, pymysql.MySQLError):
        return False


@contextmanager
def transaction(conn: Connection) -> Iterator[Connection]:
    """Execute database operations within an atomic transaction.

    Args:
        conn: An existing open MariaDB connection.

    Yields:
        The connection used for transactional operations.

    Raises:
        DatabaseConnectionError: If the database connection is lost.
        DatabaseError: If a database operation or rollback fails.
        Exception: Propagates non-database errors after rollback.
    """
    try:
        conn.begin()
        yield conn
        conn.commit()
    except Exception as exc:
        try:
            # Undo all uncommitted changes when an operation fails.
            conn.rollback()
        except pymysql.MySQLError:
            raise DatabaseError("MariaDB transaction rollback failed") from None

        if isinstance(exc, pymysql.err.OperationalError):
            raise DatabaseConnectionError("MariaDB connection failed during transaction") from None

        if isinstance(exc, pymysql.MySQLError):
            raise DatabaseError("MariaDB transaction failed") from None

        raise
