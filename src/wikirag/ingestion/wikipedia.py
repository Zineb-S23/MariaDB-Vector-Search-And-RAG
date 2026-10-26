"""Wikipedia acquisition helpers: seed titles, article fetching and snapshots."""

from __future__ import annotations

import csv
from pathlib import Path

from wikirag.errors import ValidationError

SEED_COLUMNS = ("topic", "title")


def _normalize_title(title: str) -> str:
    """Return a title with underscores as spaces and collapsed whitespace.

    Args:
        title: Raw title text from the seed file.

    Returns:
        The normalized title.
    """
    return " ".join(title.replace("_", " ").split())


def _title_id(title: str) -> str:
    """Return an identifier that is equal for titles Wikipedia treats as the same page.

    Args:
        title: A normalized title.

    Returns:
        The title with a case-insensitive first letter.
    """
    # Wikipedia titles are case-sensitive except for the first letter
    return title[:1].casefold() + title[1:]


def load_seed_titles(path: Path) -> list[tuple[str, str]]:
    """Load and validate the seed list of Wikipedia articles.

    The file is a UTF-8 CSV with the header ``topic,title``. Blank lines are skipped,
    surrounding whitespace is stripped and underscores in titles become spaces.
    Titles that contain a comma must be quoted.

    Args:
        path: Location of the seed CSV file.

    Returns:
        ``(topic, title)`` pairs in file order.

    Raises:
        ValidationError: If the file is missing or not valid UTF-8 CSV, has the wrong header,
            contains an incomplete row or a duplicate title, or has no data rows.
    """
    if not path.is_file():
        raise ValidationError(f"Seed file not found: {path}")

    seeds: list[tuple[str, str]] = []
    first_line_of: dict[str, int] = {}
    try:
        # utf-8-sig drops the byte-order mark that Excel adds
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or tuple(cell.strip() for cell in header) != SEED_COLUMNS:
                raise ValidationError(f"Seed file {path} must start with the header topic,title")
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                line_number = reader.line_num
                if len(row) != len(SEED_COLUMNS):
                    raise ValidationError(f"Line {line_number}: expected topic,title columns")
                topic = row[0].strip()
                title = _normalize_title(row[1])
                if not topic or not title:
                    raise ValidationError(f"Line {line_number}: topic and title must not be empty")
                title_id = _title_id(title)
                if title_id in first_line_of:
                    raise ValidationError(
                        f"Line {line_number}: duplicate title {title!r} "
                        f"(first seen on line {first_line_of[title_id]})"
                    )
                first_line_of[title_id] = line_number
                seeds.append((topic, title))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValidationError(f"Seed file {path} is not valid UTF-8 CSV: {error}") from error

    if not seeds:
        raise ValidationError(f"Seed file {path} contains no seed rows")
    return seeds
