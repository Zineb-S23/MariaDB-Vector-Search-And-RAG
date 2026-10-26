"""Unit tests for Experiment E measurement aggregation."""

import pytest

from scripts.run_cache_experiment import summarize_runs


def test_cache_experiment_summarizes_restart_cycles() -> None:
    """Verify latency statistics across independently collected runs.

    Returns:
        None: Passes when the aggregated measurements are correct.
    """
    runs = [
        [
            {
                "first_observed_ms": 10.0,
                "warm_durations_ms": [2.0, 4.0],
                "uses_vector_index": True,
            }
        ],
        [
            {
                "first_observed_ms": 12.0,
                "warm_durations_ms": [6.0, 8.0],
                "uses_vector_index": True,
            }
        ],
    ]

    result = summarize_runs(runs)

    assert result["restart_cycles"] == 2
    assert result["first_query_mean_ms"] == pytest.approx(11.0)
    assert result["warm_sample_count"] == 4
    assert result["warm_mean_ms"] == pytest.approx(5.0)
    assert result["warm_p50_ms"] == pytest.approx(5.0)
    assert result["uses_vector_index"] is True


@pytest.mark.parametrize("invalid_runs", [[], [[]]])
def test_cache_experiment_rejects_empty_measurements(
    invalid_runs: list[list[dict[str, object]]],
) -> None:
    """Reject missing measurements instead of publishing empty reports.

    Args:
        invalid_runs: Invalid benchmark measurements to reject.

    Returns:
        None: Passes when the invalid input raises ValueError.
    """
    with pytest.raises(ValueError):
        summarize_runs(invalid_runs)


def test_cache_experiment_reports_missing_hnsw_usage() -> None:
    """Identify runs where MariaDB did not select the HNSW index.

    Returns:
        None: Passes when missing index usage is reflected in the
            aggregated result.
    """
    runs = [
        [
            {
                "first_observed_ms": 3.0,
                "warm_durations_ms": [2.0, 3.0],
                "uses_vector_index": False,
            }
        ]
    ]

    result = summarize_runs(runs)

    assert result["uses_vector_index"] is False
