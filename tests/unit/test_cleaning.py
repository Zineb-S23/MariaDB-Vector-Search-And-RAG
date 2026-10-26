"""Unit tests for the deterministic text cleaning."""

from __future__ import annotations

import pytest

from wikirag.errors import ValidationError
from wikirag.ingestion.cleaning import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a  \t b", "a b"),
        ("a\u00a0b", "a b"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("a  \n  b", "a\nb"),
        ("a\n   \nb", "a\n\nb"),
        ("a\n\n\n\nb", "a\n\nb"),
        ("a\n\nb", "a\n\nb"),
        ("\n\n a \n\n", "a"),
        ("a\u200bb\ufeffc\u00add\u2060e", "abcde"),
        ("e\u0301", "\u00e9"),
        ("Zürich ist schön", "Zürich ist schön"),
    ],
)
def test_cleaning_rules(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_zero_width_non_joiner_is_kept() -> None:
    assert clean_text("a\u200cb") == "a\u200cb"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t \n", "\u200b\ufeff"])
def test_empty_or_whitespace_only_gives_empty_text(raw: str) -> None:
    assert clean_text(raw) == ""


def test_cleaning_is_idempotent_and_deterministic() -> None:
    messy = "\ufeff Title \r\n\r\n\r\n\r\nFirst\u00a0 paragraph.  \n\tSecond\u200b line.\n\n\n"
    once = clean_text(messy)
    assert clean_text(messy) == once
    assert clean_text(once) == once


@pytest.mark.parametrize("value", [None, 5, b"bytes", ["text"]])
def test_non_string_raises(value: object) -> None:
    with pytest.raises(ValidationError, match="string"):
        clean_text(value)  # type: ignore[arg-type]
