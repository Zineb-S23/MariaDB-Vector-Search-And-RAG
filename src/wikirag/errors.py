
"""
Shared exception types for the WikiRAG application.

All application-level errors crossing module boundaries should
inherit from WikiRagError.

This gives ingestion, database, retrieval, RAG, and the CLI
a consistent way to report and handle failures.
"""


class WikiRagError(Exception):
    """Base exception for all WikiRAG application errors."""


class ValidationError(WikiRagError):
    """Input is invalid and must be corrected before processing."""


class EmbeddingDimensionError(ValidationError):
    """An embedding has the wrong dimension or non-finite values."""


class DatabaseConnectionError(WikiRagError):
    """Unable to connect to MariaDB, or the connection was lost."""


class DatabaseError(WikiRagError):
    """A database operation failed, other than a connection failure."""


class SchemaMismatchError(WikiRagError):
    """The database schema or stored embedding metadata is incompatible."""
