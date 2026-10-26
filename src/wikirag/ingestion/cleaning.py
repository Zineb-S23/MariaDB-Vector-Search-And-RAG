"""Deterministic cleaning of Wikipedia plain text."""

from __future__ import annotations

import re
import unicodedata

from wikirag.errors import ValidationError

# zero-width space, soft hyphen, word joiner and byte-order mark carry no meaning
_INVISIBLE_CHARS = dict.fromkeys(map(ord, "\u200b\u00ad\u2060\ufeff"))
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Clean article text so that the same input always gives the same output.

    Applies these rules in order: Unicode NFC normalization, uniform ``\\n`` line endings,
    removal of invisible characters, every run of spaces, tabs and non-breaking spaces becomes
    one space, spaces at the start and end of each line are removed, three or more line breaks
    become one blank line, and the whole text is stripped. Paragraph breaks are kept.

    Args:
        text: Raw article text from Wikipedia.

    Returns:
        The cleaned text. It is empty if the input holds only whitespace.

    Raises:
        ValidationError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise ValidationError("Text to clean must be a string")

    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.translate(_INVISIBLE_CHARS)
    cleaned = _HORIZONTAL_SPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return _BLANK_LINES.sub("\n\n", cleaned).strip()
