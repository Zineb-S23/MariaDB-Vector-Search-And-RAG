import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from wikirag.errors import ValidationError

SEED_COLUMNS = ("topic", "title")
WIKIPEDIA_ARTICLE_URL = "https://en.wikipedia.org/wiki/"
CATEGORY_PREFIX = "Category:"


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


@dataclass(frozen=True, slots=True)
class RawArticle:
    """A Wikipedia article as downloaded, before cleaning and chunking.

    Attributes:
        article_id: Wikipedia page id (positive).
        title: Canonical article title.
        url: Address of the article on English Wikipedia.
        revision_id: Id of the downloaded revision.
        revised_at: Time of that revision, as naive UTC.
        text: Plain article text as returned by Wikipedia.
        fetched_at: Time of the download, as naive UTC.
        topic: Topic label from the seed list, if known.
        categories: Visible category names without the ``Category:`` prefix, sorted.
        linked_titles: Titles of linked articles, sorted and without duplicates.
    """

    article_id: int
    title: str
    url: str
    revision_id: int
    revised_at: datetime
    text: str
    fetched_at: datetime
    topic: str | None = None
    categories: tuple[str, ...] = ()
    linked_titles: tuple[str, ...] = ()


def _is_positive_int(value: object) -> bool:
    """Return whether a value is a real positive integer.

    Args:
        value: Any value to check.

    Returns:
        True for integers above zero; False for everything else, including booleans.
    """
    # bool is a subclass of int, so it has to be excluded explicitly
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_timestamp(value: object, title: str) -> datetime:
    """Convert a Wikipedia timestamp such as ``2026-09-30T08:15:00Z`` to naive UTC.

    Args:
        value: The timestamp from the API response.
        title: Article title, used in error messages.

    Returns:
        The time as a naive ``datetime`` in UTC.

    Raises:
        ValidationError: If the value is not a timestamp in the expected format.
    """
    if not isinstance(value, str):
        raise ValidationError(f"{title!r}: invalid revision timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValidationError(f"{title!r}: invalid revision timestamp {value!r}") from error


def _collect_titles(entries: Sequence[Any] | None, *, strip_prefix: str = "") -> tuple[str, ...]:
    """Collect unique, sorted titles from a list of API entries.

    Args:
        entries: List of objects that each have a ``title`` key, or None.
        strip_prefix: Text to remove from the start of every title.

    Returns:
        Sorted titles without duplicates or empty values.
    """
    names: set[str] = set()
    for entry in entries or ():
        name = entry.get("title") if isinstance(entry, Mapping) else None
        if isinstance(name, str):
            name = name.removeprefix(strip_prefix).strip()
            if name:
                names.add(name)
    # sorting makes the snapshot identical on every run
    return tuple(sorted(names))


def _article_url(title: str) -> str:
    """Build the English Wikipedia address of an article.

    Args:
        title: Article title.

    Returns:
        The full address, with spaces written as underscores.
    """
    return WIKIPEDIA_ARTICLE_URL + quote(title.replace(" ", "_"), safe="_(),'!*~")


def parse_article_page(
    page: Mapping[str, Any],
    *,
    fetched_at: datetime,
    topic: str | None = None,
) -> RawArticle:
    """Turn one page object from the Wikipedia API into a ``RawArticle``.

    Args:
        page: A page object from a Wikipedia ``action=query`` response.
        fetched_at: Time of the download, as naive UTC.
        topic: Topic label from the seed list, if known.

    Returns:
        The parsed article.

    Raises:
        ValidationError: If the page does not exist or lacks its id, revision or text.
    """
    title = page.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValidationError("Wikipedia page has no title")
    if page.get("missing") or page.get("invalid"):
        raise ValidationError(f"Wikipedia page not found: {title!r}")

    article_id = page.get("pageid")
    if not _is_positive_int(article_id):
        raise ValidationError(f"{title!r}: missing or invalid page id")

    revisions = page.get("revisions")
    if not revisions:
        raise ValidationError(f"{title!r}: no revision information")
    revision_id = revisions[0].get("revid")
    if not _is_positive_int(revision_id):
        raise ValidationError(f"{title!r}: invalid revision id")
    revised_at = _parse_timestamp(revisions[0].get("timestamp"), title)

    text = page.get("extract")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"{title!r}: article has no text")

    return RawArticle(
        article_id=article_id,
        title=title,
        url=_article_url(title),
        revision_id=revision_id,
        revised_at=revised_at,
        text=text,
        fetched_at=fetched_at,
        topic=topic,
        categories=_collect_titles(page.get("categories"), strip_prefix=CATEGORY_PREFIX),
        linked_titles=_collect_titles(page.get("links")),
    )
