"""
Shared pytest fixtures for the WikiRAG test suite.

Fixtures provide reusable, isolated test components so that unit tests
do not depend on MariaDB or a real embedding model.

Database-specific integration fixtures will be introduced separately.
"""

from collections.abc import Iterator

import pytest
from fakes import FakeEmbedder, InMemoryRepository


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """
    Create a fresh deterministic embedding model for a test.

    Returns:
        FakeEmbedder: An isolated fake embedding model producing
            normalized, 384-dimensional vectors.
    """

    return FakeEmbedder()


@pytest.fixture
def fake_repo() -> Iterator[InMemoryRepository]:
    """
    Create an empty in-memory repository for a test.

    Each test receives its own repository instance, preventing
    stored articles and chunks from leaking between tests.

    Yields:
        InMemoryRepository: An empty repository that implements the
            shared database interface.
    """

    repo = InMemoryRepository()

    try:
        yield repo
    finally:
        # Close the repository even when a test fails.
        repo.close()
