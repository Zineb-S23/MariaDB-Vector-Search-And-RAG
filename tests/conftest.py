"""
Shared pytest fixtures for the WikiRAG test suite.

Fixtures provide reusable, isolated test components so that unit tests
do not depend on MariaDB or a real embedding model.

Database-specific integration fixtures will be introduced separately.
"""

from collections.abc import Iterator
from pathlib import Path

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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically categorize tests based on their directory.

    Pytest calls this hook after discovering tests.

    Args:
        items (list[pytest.Item]): Tests collected by pytest.

    Returns:
        None: Adds a category marker to each matching test.
    """
    tests_root = Path(__file__).resolve().parent

    categories = {
        "unit": pytest.mark.unit,
        "integration": pytest.mark.integration,
        "e2e": pytest.mark.e2e,
    }

    for item in items:
        try:
            relative_path = item.path.relative_to(tests_root)
        except ValueError:
            # Ignore tests that are outside our project's tests directory.
            continue

        if relative_path.parts:
            category = relative_path.parts[0]

            if category in categories:
                item.add_marker(categories[category])
