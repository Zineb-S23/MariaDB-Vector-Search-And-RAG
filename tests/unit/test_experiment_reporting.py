"""Unit tests for reusable WikiRAG experiment report writing."""

import json
from pathlib import Path

import pytest

from wikirag.errors import ValidationError
from wikirag.experiments.reporting import write_experiment_report


def test_write_experiment_report_creates_json(
    tmp_path: Path,
) -> None:
    """Save an experiment report and preserve its structured values.

    Args:
        tmp_path: Pytest-managed temporary output directory.
    """
    report = {
        "experiment": "metric_mismatch",
        "mariadb_version": "11.8.9",
        "repeats": 5,
        "variants": {
            "cosine": {
                "uses_vector_index": True,
                "durations_ms": [0.2, 0.3],
            },
            "euclidean": {
                "uses_vector_index": False,
                "durations_ms": [0.4, 0.5],
            },
        },
    }

    destination = tmp_path / "reports" / "metric_mismatch.json"

    result = write_experiment_report(
        report,
        output_path=destination,
    )

    assert result == destination
    assert destination.is_file()

    saved_report = json.loads(destination.read_text(encoding="utf-8"))

    assert saved_report == report


@pytest.mark.parametrize(
    "invalid_report",
    (
        {},
        {"measurement": object()},
        {"measurement": float("nan")},
    ),
)
def test_write_experiment_report_rejects_invalid_data(
    tmp_path: Path,
    invalid_report: dict[str, object],
) -> None:
    """Reject missing or non-JSON-compatible experiment values.

    Args:
        tmp_path: Pytest-managed temporary output directory.
        invalid_report: Report containing unsupported values.
    """
    destination = tmp_path / "invalid.json"

    with pytest.raises(ValidationError):
        write_experiment_report(
            invalid_report,
            output_path=destination,
        )

    assert not destination.exists()
