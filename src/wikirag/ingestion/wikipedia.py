"""Wikipedia acquisition helpers: seed titles, article fetching and snapshots."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from wikirag.errors import ValidationError, WikiRagError

SEED_COLUMNS = ("topic", "title")
WIKIPEDIA_ARTICLE_URL = "https://en.wikipedia.org/wiki/"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "WikiRAG-student-project/0.1 (https://github.com/melsayed-eg/MariaDB-Vector-Search-And-RAG)"
)
CATEGORY_PREFIX = "Category:"
REQUEST_TIMEOUT_S = 30.0
MAX_API_REQUESTS = 20


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


def _build_query_params(title: str) -> dict[str, str]:
    """Build the parameters of the first API request for one article.

    Args:
        title: Normalized article title.

    Returns:
        Query parameters asking for plain text, revision info, visible categories and links.
    """
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "titles": title,
        "redirects": "1",
        "prop": "extracts|revisions|categories|links",
        "explaintext": "1",
        "exsectionformat": "plain",
        "rvprop": "ids|timestamp",
        "clshow": "!hidden",
        "cllimit": "max",
        "plnamespace": "0",
        "pllimit": "max",
    }


def _get_json(
    client: requests.Session, params: Mapping[str, str], timeout_s: float
) -> dict[str, Any]:
    """Send one request to the Wikipedia API and return the decoded answer.

    Args:
        client: HTTP session used to send the request.
        params: Query parameters of the request.
        timeout_s: Seconds to wait before giving up.

    Returns:
        The decoded JSON object.

    Raises:
        WikiRagError: If the request fails, the server reports an error or the answer is
            not a JSON object.
    """
    try:
        response = client.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise WikiRagError(f"Wikipedia request failed: {error}") from error
    if not isinstance(payload, dict):
        raise WikiRagError("Wikipedia returned an unexpected response")
    if "error" in payload:
        raise WikiRagError(f"Wikipedia API error: {payload['error']}")
    return payload


def _merge_page(merged: dict[str, Any], page: Mapping[str, Any]) -> None:
    """Add the fields of one API answer to the article collected so far.

    Args:
        merged: The article collected from earlier answers; changed in place.
        page: The page object of the newest answer.
    """
    for key, value in page.items():
        if key in ("categories", "links"):
            # long lists arrive in several answers, so they are joined
            merged.setdefault(key, []).extend(value)
        else:
            merged.setdefault(key, value)


def fetch_article(
    title: str,
    *,
    topic: str | None = None,
    session: requests.Session | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
    max_requests: int = MAX_API_REQUESTS,
) -> RawArticle:
    """Download one article from English Wikipedia.

    Follows redirects, requests the plain text, the latest revision, visible categories and
    article links, and keeps asking while Wikipedia says there is more to come.

    Args:
        title: Article title; underscores and extra spaces are tolerated.
        topic: Topic label from the seed list, stored on the result.
        session: HTTP session to reuse; a temporary one is created and closed if omitted.
        timeout_s: Seconds to wait for each request.
        max_requests: Most requests allowed for this article, as a safety stop.

    Returns:
        The downloaded article.

    Raises:
        ValidationError: If the title is blank, ``max_requests`` is below 1, or the page
            does not exist or lacks required data.
        WikiRagError: If Wikipedia cannot be reached, reports an error or keeps asking for
            more requests than ``max_requests``.
    """
    clean_title = _normalize_title(title)
    if not clean_title:
        raise ValidationError("Article title must not be empty")
    if max_requests < 1:
        raise ValidationError("max_requests must be at least 1")

    client = session if session is not None else requests.Session()
    merged: dict[str, Any] = {}
    base_params = _build_query_params(clean_title)
    params = base_params
    try:
        for _ in range(max_requests):
            payload = _get_json(client, params, timeout_s)
            pages = payload.get("query", {}).get("pages")
            if not pages:
                raise WikiRagError(f"Wikipedia returned no page for {clean_title!r}")
            _merge_page(merged, pages[0])
            continuation = payload.get("continue")
            if not continuation:
                break
            # the continue token tells Wikipedia where the last answer stopped
            params = {**base_params, **continuation}
        else:
            raise WikiRagError(f"Too many API requests for {clean_title!r}")
    finally:
        if session is None:
            client.close()

    fetched_at = datetime.now(UTC).replace(tzinfo=None)
    return parse_article_page(merged, fetched_at=fetched_at, topic=topic)
