"""Verify the real repository exposes the planned search interface."""

import inspect

import pytest
from fakes import InMemoryRepository

from wikirag.db.repository import WikiRepository


@pytest.mark.parametrize(
    "method_name",
    (
        "search_chunks",
        "search_chunks_exact",
        "set_ef_search",
    ),
)
def test_search_method_signatures_match_fake(method_name: str) -> None:
    """Check search method signatures match the shared fake repository.

    Args:
        method_name: Public repository method to inspect.
    """
    real_method = getattr(WikiRepository, method_name)
    fake_method = getattr(InMemoryRepository, method_name)

    assert inspect.signature(real_method) == inspect.signature(fake_method)
