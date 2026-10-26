"""Wikipedia acquisition helpers: seed titles, article fetching and snapshots."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
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
SNAPSHOT_FIELDS = (
    "article_id",
    "title",
    "url",
    "revision_id",
    "revised_at",
    "fetched_at",
    "topic",
    "categories",
    "linked_titles",
    "text",
)

logger = logging.getLogger(__name__)


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


def _article_to_record(article: RawArticle) -> dict[str, Any]:
    """Convert an article into a JSON-friendly dictionary.

    Args:
        article: The article to store.

    Returns:
        A dictionary with every field of the article; times as ``YYYY-MM-DDTHH:MM:SSZ`` text.
    """
    return {
        "article_id": article.article_id,
        "title": article.title,
        "url": article.url,
        "revision_id": article.revision_id,
        "revised_at": article.revised_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_at": article.fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "topic": article.topic,
        "categories": list(article.categories),
        "linked_titles": list(article.linked_titles),
        "text": article.text,
    }


def _record_to_article(record: object) -> RawArticle:
    """Rebuild an article from a dictionary read from a snapshot line.

    Args:
        record: The decoded JSON value of one snapshot line.

    Returns:
        The validated article.

    Raises:
        ValidationError: If the record is not an object, lacks a field or holds a bad value.
    """
    if not isinstance(record, dict):
        raise ValidationError("expected a JSON object")
    missing = [field for field in SNAPSHOT_FIELDS if field not in record]
    if missing:
        raise ValidationError(f"missing fields: {', '.join(missing)}")
    for field in ("title", "url", "text"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{field} must be a non-empty string")
    for field in ("article_id", "revision_id"):
        if not _is_positive_int(record[field]):
            raise ValidationError(f"{field} must be a positive integer")
    topic = record["topic"]
    if topic is not None and not isinstance(topic, str):
        raise ValidationError("topic must be a string or null")
    for field in ("categories", "linked_titles"):
        value = record[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValidationError(f"{field} must be a list of strings")

    title = record["title"]
    return RawArticle(
        article_id=record["article_id"],
        title=title,
        url=record["url"],
        revision_id=record["revision_id"],
        revised_at=_parse_timestamp(record["revised_at"], title),
        text=record["text"],
        fetched_at=_parse_timestamp(record["fetched_at"], title),
        topic=topic,
        categories=tuple(record["categories"]),
        linked_titles=tuple(record["linked_titles"]),
    )


def _iter_snapshot(path: Path) -> Iterator[RawArticle]:
    """Yield the articles of a snapshot file one by one.

    Args:
        path: An existing snapshot file.

    Yields:
        Each article, in file order.

    Raises:
        ValidationError: If a line is not valid JSON, holds an invalid article, or the file
            is not UTF-8.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    article = _record_to_article(json.loads(line))
                except (json.JSONDecodeError, ValidationError) as error:
                    raise ValidationError(f"Snapshot line {line_number}: {error}") from error
                yield article
    except UnicodeDecodeError as error:
        raise ValidationError(f"Snapshot file {path} is not valid UTF-8: {error}") from error


def load_snapshot(path: Path) -> Iterator[RawArticle]:
    """Read a snapshot file written by ``build_snapshot``.

    The file is checked right away, but the articles are read lazily, so a large snapshot
    never has to fit into memory at once.

    Args:
        path: Location of the snapshot file (one JSON article per line).

    Returns:
        An iterator over the articles in file order.

    Raises:
        ValidationError: If the file does not exist. Damaged lines raise the same error
            while the iterator is consumed.
    """
    if not path.is_file():
        raise ValidationError(f"Snapshot file not found: {path}")
    return _iter_snapshot(path)


def build_snapshot(
    seed_path: Path,
    out_path: Path,
    *,
    delay_s: float = 0.2,
) -> int:
    """Download every seed article from Wikipedia and save them as a snapshot.

    Articles that cannot be downloaded are skipped with a warning, and so is a second seed
    title that leads to an article already saved (for example through a redirect). The file
    is written to a temporary name first, so an existing snapshot is only replaced when at
    least one article was downloaded.

    Args:
        seed_path: Seed CSV file with ``topic,title`` columns.
        out_path: Snapshot file to write, one JSON article per line.
        delay_s: Seconds to wait between two requests, to be polite to Wikipedia.

    Returns:
        The number of articles written.

    Raises:
        ValidationError: If ``delay_s`` is negative or the seed file is invalid.
        WikiRagError: If no article at all could be downloaded.
    """
    if delay_s < 0:
        raise ValidationError("delay_s must not be negative")
    seeds = load_seed_titles(seed_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(out_path.name + ".tmp")
    seen_ids: set[int] = set()
    written = 0
    try:
        with requests.Session() as client, temp_path.open("w", encoding="utf-8") as handle:
            for index, (topic, title) in enumerate(seeds):
                if index > 0 and delay_s > 0:
                    time.sleep(delay_s)
                try:
                    article = fetch_article(title, topic=topic, session=client)
                except WikiRagError as error:
                    logger.warning("Skipping %r: %s", title, error)
                    continue
                if article.article_id in seen_ids:
                    logger.warning("Skipping %r: article already in the snapshot", title)
                    continue
                seen_ids.add(article.article_id)
                # sorted keys keep the file identical when the data is identical
                line = json.dumps(_article_to_record(article), ensure_ascii=False, sort_keys=True)
                handle.write(line + "\n")
                written += 1
        if written == 0:
            raise WikiRagError("No articles could be downloaded; snapshot not written")
        temp_path.replace(out_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return written
