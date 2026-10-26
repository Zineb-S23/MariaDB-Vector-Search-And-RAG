"""Unit tests for turning Wikipedia API pages into RawArticle objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import Any

import pytest

from wikirag.errors import ValidationError
from wikirag.ingestion.wikipedia import RawArticle, parse_article_page

FETCHED_AT = datetime(2026, 10, 25, 12, 0, 0)


def make_page(**overrides: Any) -> dict[str, Any]:
    """Build a realistic API page and apply any overrides."""
    page: dict[str, Any] = {
        "pageid": 4223,
        "title": "Black hole",
        "extract": "A black hole is a region of spacetime.",
        "revisions": [{"revid": 123456, "timestamp": "2026-09-30T08:15:00Z"}],
        "categories": [
            {"ns": 14, "title": "Category:Black holes"},
            {"ns": 14, "title": "Category:Astrophysics"},
        ],
        "links": [{"ns": 0, "title": "Star"}, {"ns": 0, "title": "Event horizon"}],
    }
    page.update(overrides)
    return page


def test_builds_article_from_page() -> None:
    article = parse_article_page(make_page(), fetched_at=FETCHED_AT, topic="physics")
    assert article == RawArticle(
        article_id=4223,
        title="Black hole",
        url="https://en.wikipedia.org/wiki/Black_hole",
        revision_id=123456,
        revised_at=datetime(2026, 9, 30, 8, 15, 0),
        text="A black hole is a region of spacetime.",
        fetched_at=FETCHED_AT,
        topic="physics",
        categories=("Astrophysics", "Black holes"),
        linked_titles=("Event horizon", "Star"),
    )


def test_topic_defaults_to_none() -> None:
    assert parse_article_page(make_page(), fetched_at=FETCHED_AT).topic is None


def test_url_uses_underscores_for_spaces() -> None:
    page = make_page(title="Hans Zimmer")
    assert parse_article_page(page, fetched_at=FETCHED_AT).url.endswith("/wiki/Hans_Zimmer")


def test_missing_categories_and_links_become_empty_tuples() -> None:
    page = make_page()
    del page["categories"]
    del page["links"]
    article = parse_article_page(page, fetched_at=FETCHED_AT)
    assert article.categories == ()
    assert article.linked_titles == ()


def test_duplicate_and_empty_titles_are_dropped() -> None:
    page = make_page(
        links=[{"title": "Star"}, {"title": "Star"}, {"title": "  "}, {"ns": 0}, "oops"],
    )
    assert parse_article_page(page, fetched_at=FETCHED_AT).linked_titles == ("Star",)


def test_missing_page_raises() -> None:
    with pytest.raises(ValidationError, match="not found"):
        parse_article_page({"title": "Nope", "missing": True}, fetched_at=FETCHED_AT)


def test_page_without_title_raises() -> None:
    with pytest.raises(ValidationError, match="no title"):
        parse_article_page(make_page(title=""), fetched_at=FETCHED_AT)


@pytest.mark.parametrize("bad_id", [None, 0, -5, "4223", True])
def test_invalid_page_id_raises(bad_id: object) -> None:
    with pytest.raises(ValidationError, match="page id"):
        parse_article_page(make_page(pageid=bad_id), fetched_at=FETCHED_AT)


@pytest.mark.parametrize(
    "revisions",
    [None, [], [{"timestamp": "2026-09-30T08:15:00Z"}], [{"revid": 0, "timestamp": "x"}]],
)
def test_missing_or_invalid_revision_raises(revisions: object) -> None:
    with pytest.raises(ValidationError, match="revision"):
        parse_article_page(make_page(revisions=revisions), fetched_at=FETCHED_AT)


@pytest.mark.parametrize("timestamp", [None, "yesterday", "2026-09-30"])
def test_invalid_timestamp_raises(timestamp: object) -> None:
    page = make_page(revisions=[{"revid": 1, "timestamp": timestamp}])
    with pytest.raises(ValidationError, match="timestamp"):
        parse_article_page(page, fetched_at=FETCHED_AT)


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_empty_text_raises(text: object) -> None:
    with pytest.raises(ValidationError, match="no text"):
        parse_article_page(make_page(extract=text), fetched_at=FETCHED_AT)


def test_raw_article_cannot_be_changed() -> None:
    article = parse_article_page(make_page(), fetched_at=FETCHED_AT)
    with pytest.raises(FrozenInstanceError):
        article.title = "Something else"  # type: ignore[misc]
