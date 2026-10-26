"""Write reproducible WikiRAG experiment reports to JSON files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from wikirag.errors import ValidationError


def write_experiment_report(
    report: Mapping[str, object],
    *,
    output_path: Path,
) -> Path:
    """Save a structured experiment report as formatted JSON.

    Args:
        report: Non-empty mapping containing experiment metadata,
            execution plans, measurements, and observations.
        output_path: Destination JSON file.

    Returns:
        Path: Location of the written report.

    Raises:
        ValidationError: If the report is empty, cannot be encoded
            as JSON, or the output path is invalid.
        OSError: If the destination cannot be created or written.
    """
    if not isinstance(report, Mapping) or not report:
        raise ValidationError("Experiment report must be a non-empty mapping")

    if not isinstance(output_path, Path) or output_path.suffix.lower() != ".json":
        raise ValidationError("output_path must be a Path ending in .json")

    try:
        payload = json.dumps(
            dict(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ValidationError("Experiment report must contain JSON-serializable values") from None

    # Serialize before creating the output directory or writing the file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")

    return output_path
