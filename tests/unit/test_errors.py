"""
Unit tests for the shared WikiRAG exception hierarchy.

These tests verify that every application exception inherits from
WikiRagError and that EmbeddingDimensionError is a specialized
ValidationError.

No database connection or external service is required.
"""

import pytest

from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    EmbeddingDimensionError,
    SchemaMismatchError,
    ValidationError,
    WikiRagError,
)


@pytest.mark.parametrize(
    "exception_class",
    [
        ValidationError,
        EmbeddingDimensionError,
        DatabaseConnectionError,
        DatabaseError,
        SchemaMismatchError,
    ],
)
def test_exceptions_inherit_from_wikirag_error(
    exception_class: type[Exception],
) -> None:
    """
    Verify that all application exceptions share the same base class.

    Args:
        exception_class (type[Exception]): Exception class supplied
            by pytest for inheritance testing.

    Returns:
        None: The test passes when inheritance is correct.

    Raises:
        AssertionError: If an exception does not inherit from
            WikiRagError.
    """

    assert issubclass(exception_class, WikiRagError)


def test_embedding_dimension_error_inherits_from_validation_error() -> None:
    """
    Verify that invalid embedding dimensions are validation errors.

    Returns:
        None: The test passes when EmbeddingDimensionError inherits
            from ValidationError.

    Raises:
        AssertionError: If the inheritance hierarchy is incorrect.
    """

    assert issubclass(EmbeddingDimensionError, ValidationError)


@pytest.mark.parametrize(
    "exception_class",
    [
        WikiRagError,
        ValidationError,
        EmbeddingDimensionError,
        DatabaseConnectionError,
        DatabaseError,
        SchemaMismatchError,
    ],
)
def test_exceptions_preserve_error_messages(
    exception_class: type[Exception],
) -> None:
    """
    Verify that application exceptions preserve diagnostic messages.

    Args:
        exception_class (type[Exception]): Exception class being tested.

    Returns:
        None: The test passes when the original message is preserved.

    Raises:
        AssertionError: If the exception message changes unexpectedly.
    """

    message = "Example application error"

    error = exception_class(message)

    assert str(error) == message


def test_validation_error_can_be_caught_as_base_error() -> None:
    """
    Verify that a validation failure can be handled as WikiRagError.

    Returns:
        None: The test passes when the base exception catches
            the validation failure.

    Raises:
        AssertionError: If the exception cannot be caught correctly.
    """

    # A common base class lets the CLI handle application errors consistently.
    with pytest.raises(WikiRagError, match="Invalid input"):
        raise ValidationError("Invalid input")


def test_embedding_error_can_be_caught_as_validation_error() -> None:
    """
    Verify that embedding failures can be caught as validation errors.

    Returns:
        None: The test passes when ValidationError catches an
            EmbeddingDimensionError.

    Raises:
        AssertionError: If the specialized exception is not caught.
    """

    with pytest.raises(ValidationError, match="Expected 384 dimensions"):
        raise EmbeddingDimensionError("Expected 384 dimensions, received 256")


def test_database_errors_have_distinct_types() -> None:
    """
    Verify that database failure categories remain distinguishable.

    Returns:
        None: The test passes when all three database-related
            exception classes are distinct.

    Raises:
        AssertionError: If two exception classes are identical.
    """

    exception_classes = {
        DatabaseConnectionError,
        DatabaseError,
        SchemaMismatchError,
    }

    # Each error category must remain identifiable by its exception type.
    assert len(exception_classes) == 3


def test_database_connection_error_is_not_database_error() -> None:
    """
    Verify that connection failures are distinct from SQL failures.

    Returns:
        None: The test passes when a connection failure is not
            incorrectly classified as DatabaseError.

    Raises:
        AssertionError: If the exception hierarchy is incorrect.
    """

    error = DatabaseConnectionError("Connection failed")

    assert isinstance(error, WikiRagError)
    assert not isinstance(error, DatabaseError)
