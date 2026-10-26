"""Experiment D: MariaDB transaction and vector visibility."""

from __future__ import annotations

from wikirag.config import DbConfig
from wikirag.db.diagnostics import demo_transaction_visibility


def run_transaction_visibility(
    config: DbConfig | None = None,
) -> dict[str, object]:
    """Run transaction visibility experiments at two isolation levels.

    Args:
        config: Dedicated MariaDB test database configuration.

    Returns:
        dict[str, object]: Observed relational and vector-search
            visibility before commit, after rollback, after commit,
            and after cleanup.

    Raises:
        ValidationError: If a non-test database is selected.
        DatabaseConnectionError: If MariaDB is unavailable.
        DatabaseError: If the experiment's SQL execution fails.
    """
    return demo_transaction_visibility(config)
