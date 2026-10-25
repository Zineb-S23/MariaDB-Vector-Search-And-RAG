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


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    (
        ("search_chunks_exact", ((1.0,),)),
        ("set_ef_search", (20,)),
    ),
)
def test_search_stubs_raise_not_implemented(
    method_name: str,
    arguments: tuple[object, ...],
) -> None:
    """Confirm unfinished methods fail explicitly rather than silently.

    Args:
        method_name: Public search method being tested.
        arguments: Positional arguments supplied to the stub.
    """
    # The stubs do not access MariaDB, so no connection is required.
    repo = object.__new__(WikiRepository)

    with pytest.raises(NotImplementedError, match="feature/database-search"):
        getattr(repo, method_name)(*arguments)
