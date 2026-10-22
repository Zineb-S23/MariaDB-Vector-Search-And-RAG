
"""
Centralized configuration for the MariaDB WikiRAG application.

This module provides validated, immutable configuration objects shared
by the database, ingestion, retrieval, RAG, and evaluation components.

Configuration is loaded from environment variables and the project's
private .env file. Credentials must never be hardcoded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from wikirag.errors import ValidationError

# ---------------------------------------------------------------------
# Shared embedding constants
# ---------------------------------------------------------------------

# All teammates must use the same embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# This dimension must match the VECTOR(384) column in MariaDB.
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DbConfig:
    """
    Store immutable MariaDB connection settings.

    Attributes:
        host (str): Hostname or IP address of the database server.
        port (int): TCP port exposed by the database server.
        database (str): Name of the application database.
        user (str): Username used for database authentication.
        password (str): Password for the database user. Excluded
            from the automatically generated object representation.
    """

    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)


# ---------------------------------------------------------------------
# Application-wide configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Store immutable application settings shared by all components.

    Attributes:
        db (DbConfig): Connection settings for the main database.
        db_name_test (str): Name of the separate integration-test database.
        embedding_model (str): Sentence Transformer model identifier.
        log_level (str): Application logging level.
        llm_provider (str): Optional LLM provider; defaults to "none".
        llm_model (str | None): Optional LLM model identifier.
        llm_api_key (str | None): Optional API key, hidden from repr().
        llm_base_url (str | None): Optional LLM service URL.
    """

    db: DbConfig
    db_name_test: str
    embedding_model: str
    log_level: str

    # LLM configuration remains optional because extractive RAG is the default.
    llm_provider: str = "none"
    llm_model: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str | None = None


# ---------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------


def _required_env(name: str) -> str:
    """
    Read and validate a required environment variable.

    Args:
        name (str): Name of the environment variable to retrieve.

    Returns:
        str: The environment variable's value with surrounding
        whitespace removed.

    Raises:
        ValidationError: If the variable is missing or contains
            only whitespace.
    """

    value = os.getenv(name)

    # Reject missing and blank values before attempting database access.
    if value is None or not value.strip():
        raise ValidationError(
            f"Missing or empty required environment variable: {name}"
        )

    return value.strip()


def _optional_env(name: str) -> str | None:
    """
    Read an optional environment variable.

    Args:
        name (str): Name of the environment variable to retrieve.

    Returns:
        str | None: The stripped value when present and non-empty;
            otherwise None.
    """

    value = os.getenv(name)

    if value is None:
        return None

    # Represent missing optional configuration consistently with None.
    return value.strip() or None


def _read_port() -> int:
    """
    Read and validate the MariaDB connection port.

    The default port is 13307, matching our Docker Compose development
    setup. An explicit DB_PORT environment variable overrides it.

    Returns:
        int: A valid TCP port number between 1 and 65535.

    Raises:
        ValidationError: If DB_PORT is not an integer or is outside
            the valid TCP port range.
    """

    raw_port = os.getenv("DB_PORT", "13307")

    try:
        port = int(raw_port)

    except ValueError as exc:
        raise ValidationError(
            "DB_PORT must be a valid integer"
        ) from exc

    # TCP port numbers must fall within this valid range.
    if not 1 <= port <= 65535:
        raise ValidationError(
            "DB_PORT must be between 1 and 65535"
        )

    return port


# ---------------------------------------------------------------------
# Public configuration loader
# ---------------------------------------------------------------------


def load_settings() -> Settings:
    """
    Load and validate the application's configuration.

    The private .env file is located relative to this module, so
    configuration loading works regardless of the current working
    directory.

    Existing environment variables take precedence over .env values.
    This allows GitHub Actions and other deployment environments to
    inject their own configuration.

    Returns:
        Settings: An immutable Settings object containing validated
            database, test, embedding, logging, and optional LLM settings.

    Raises:
        ValidationError: If required database credentials are missing,
            the database port is invalid, test database naming is
            unsafe, or other configuration values are invalid.
    """

    # Resolve the repository root from src/wikirag/config.py.
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    # Do not override environment variables supplied by the operating system.
    load_dotenv(dotenv_path=env_path, override=False)

    # Validate and assemble the main database configuration.
    db = DbConfig(
        host=_required_env("DB_HOST"),
        port=_read_port(),
        database=_required_env("DB_NAME"),
        user=_required_env("DB_USER"),
        password=_required_env("DB_PASSWORD"),
    )

    # Test operations must never target the main application database.
    db_name_test = os.getenv("DB_NAME_TEST", "wikirag_test").strip()

    if (
        not db_name_test.endswith("_test")
        or db_name_test == db.database
    ):
        raise ValidationError(
            "DB_NAME_TEST must end with '_test' "
            "and differ from DB_NAME"
        )

    # Both ingestion and retrieval must use the same embedding model.
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        EMBEDDING_MODEL_NAME,
    ).strip()

    if not embedding_model:
        raise ValidationError(
            "EMBEDDING_MODEL cannot be empty"
        )

    # Default to INFO rather than enabling verbose debugging automatically.
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    valid_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if log_level not in valid_log_levels:
        raise ValidationError(
            "LOG_LEVEL must be one of: DEBUG, INFO, "
            "WARNING, ERROR, CRITICAL"
        )

    # The default generator works offline without an external LLM service.
    llm_provider = os.getenv(
        "LLM_PROVIDER",
        "none",
    ).strip().lower()

    if not llm_provider:
        raise ValidationError(
            "LLM_PROVIDER cannot be empty"
        )

    # Return one consistent configuration object for the entire project.
    return Settings(
        db=db,
        db_name_test=db_name_test,
        embedding_model=embedding_model,
        log_level=log_level,
        llm_provider=llm_provider,
        llm_model=_optional_env("LLM_MODEL"),
        llm_api_key=_optional_env("LLM_API_KEY"),
        llm_base_url=_optional_env("LLM_BASE_URL"),
    )
