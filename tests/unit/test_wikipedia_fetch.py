"""Unit tests for fetching articles from the Wikipedia API without any network access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests

from wikirag.errors import ValidationError, WikiRagError
from wikirag.ingestion.wikipedia import USER_AGENT, fetch_article


class FakeResponse:
    """A stand-in for an HTTP response."""

    def __init__(
        self, payload: Any = None, *, status_error: bool = False, bad_json: bool = False
    ) -> None:
        self.payload = payload
        self.status_error = status_error
        self.bad_json = bad_json

    def raise_for_status(self) -> None:
        if self.status_error:
            raise requests.HTTPError("503 Server Error")

    def json(self) -> Any:
        if self.bad_json:
            raise ValueError("not json")
        return self.payload


class FakeSession(requests.Session):
    """A session that returns prepared answers and records every request."""

    def __init__(
        self, responses: list[FakeResponse] | None = None, *, error: Exception | None = None
    ) -> None:
        super().__init__()
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


def page_payload(**overrides: Any) -> dict[str, Any]:
    """Build a realistic API answer for one page and apply any overrides."""
    page: dict[str, Any] = {
        "pageid": 4223,
        "title": "Black hole",
        "extract": "A black hole is a region of spacetime.",
        "revisions": [{"revid": 123456, "timestamp": "2026-09-30T08:15:00Z"}],
        "categories": [{"title": "Category:Black holes"}],
        "links": [{"title": "Star"}],
    }
    page.update(overrides)
    return {"query": {"pages": [page]}}


def test_fetches_and_parses_article() -> None:
    session = FakeSession([FakeResponse(page_payload())])
    article = fetch_article("Black hole", topic="physics", session=session)
    assert article.article_id == 4223
    assert article.title == "Black hole"
    assert article.topic == "physics"
    assert article.categories == ("Black holes",)
    assert article.linked_titles == ("Star",)


def test_fetched_at_is_recent_naive_utc() -> None:
    session = FakeSession([FakeResponse(page_payload())])
    article = fetch_article("Black hole", session=session)
    now = datetime.now(UTC).replace(tzinfo=None)
    assert article.fetched_at.tzinfo is None
    assert abs(now - article.fetched_at) < timedelta(minutes=1)


def test_request_carries_title_user_agent_and_redirects() -> None:
    session = FakeSession([FakeResponse(page_payload())])
    fetch_article("Black_hole", session=session, timeout_s=5.0)
    call = session.calls[0]
    assert call["params"]["titles"] == "Black hole"
    assert call["params"]["redirects"] == "1"
    assert call["headers"]["User-Agent"] == USER_AGENT
    assert call["timeout"] == 5.0


def test_continuation_joins_long_lists() -> None:
    first = page_payload()
    first["continue"] = {"plcontinue": "4223|0|Zebra", "continue": "||"}
    second = {
        "query": {
            "pages": [
                {"pageid": 4223, "title": "Black hole", "links": [{"title": "Event horizon"}]}
            ]
        }
    }
    session = FakeSession([FakeResponse(first), FakeResponse(second)])
    article = fetch_article("Black hole", session=session)
    assert article.linked_titles == ("Event horizon", "Star")
    assert "plcontinue" not in session.calls[0]["params"]
    assert session.calls[1]["params"]["plcontinue"] == "4223|0|Zebra"


def test_too_many_continuations_raises() -> None:
    endless = page_payload()
    endless["continue"] = {"plcontinue": "x", "continue": "||"}
    session = FakeSession([FakeResponse(endless) for _ in range(5)])
    with pytest.raises(WikiRagError, match="Too many"):
        fetch_article("Black hole", session=session, max_requests=3)
    assert len(session.calls) == 3


def test_missing_page_raises_validation_error() -> None:
    payload = {"query": {"pages": [{"title": "Nope", "missing": True}]}}
    session = FakeSession([FakeResponse(payload)])
    with pytest.raises(ValidationError, match="not found"):
        fetch_article("Nope", session=session)


@pytest.mark.parametrize("title", ["", "   ", "___"])
def test_blank_title_raises_without_any_request(title: str) -> None:
    session = FakeSession()
    with pytest.raises(ValidationError, match="empty"):
        fetch_article(title, session=session)
    assert session.calls == []


def test_invalid_max_requests_raises() -> None:
    with pytest.raises(ValidationError, match="max_requests"):
        fetch_article("Black hole", session=FakeSession(), max_requests=0)


def test_http_error_becomes_project_error() -> None:
    session = FakeSession([FakeResponse(status_error=True)])
    with pytest.raises(WikiRagError, match="request failed"):
        fetch_article("Black hole", session=session)


def test_connection_error_becomes_project_error() -> None:
    session = FakeSession(error=requests.ConnectionError("no internet"))
    with pytest.raises(WikiRagError, match="request failed"):
        fetch_article("Black hole", session=session)


def test_invalid_json_becomes_project_error() -> None:
    session = FakeSession([FakeResponse(bad_json=True)])
    with pytest.raises(WikiRagError, match="request failed"):
        fetch_article("Black hole", session=session)


def test_api_error_payload_becomes_project_error() -> None:
    session = FakeSession([FakeResponse({"error": {"code": "ratelimited"}})])
    with pytest.raises(WikiRagError, match="API error"):
        fetch_article("Black hole", session=session)


def test_answer_without_pages_raises() -> None:
    session = FakeSession([FakeResponse({"query": {}})])
    with pytest.raises(WikiRagError, match="no page"):
        fetch_article("Black hole", session=session)
