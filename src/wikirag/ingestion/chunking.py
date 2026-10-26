"""Deterministic splitting of cleaned article text into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass

from wikirag.errors import ValidationError

# preferred places to cut, strongest first
_BREAKS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A piece of article text before it is embedded.

    Attributes:
        chunk_index: Zero-based position of the chunk within its article.
        text: The chunk text, without leading or trailing whitespace.
        section_title: Title of the section the chunk belongs to, if known.
    """

    chunk_index: int
    text: str
    section_title: str | None = None


def _is_int(value: object) -> bool:
    """Return whether a value is an integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _find_break(text: str, low: int, high: int) -> int:
    """Find the best place to end a chunk inside ``text[low:high]``.

    Args:
        text: The full article text.
        low: Earliest allowed end position.
        high: Latest allowed end position (the hard limit).

    Returns:
        The end position after the strongest separator found, or ``high`` if none exists.
    """
    for separator in _BREAKS:
        position = text.rfind(separator, low, high)
        if position != -1:
            return position + len(separator)
    return high


def _next_start(text: str, start: int, end: int, overlap_chars: int) -> int:
    """Compute where the next chunk begins so that it overlaps the previous one.

    Args:
        text: The full article text.
        start: Start position of the chunk just made.
        end: End position of the chunk just made.
        overlap_chars: Wanted overlap in characters.

    Returns:
        A position after ``start``, moved forward to the beginning of a word where possible.
    """
    # always move forward, even for a tiny window, so the loop cannot stall
    position = max(end - overlap_chars, start + 1)
    while position < end and not text[position - 1].isspace():
        position += 1
    return position


def split_into_chunks(
    text: str,
    *,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[ChunkDraft]:
    """Split cleaned text into overlapping chunks of at most ``max_chars`` characters.

    Each chunk ends at a paragraph break, line break, sentence end or space found in the
    second half of its window; otherwise it is cut at ``max_chars``. The next chunk starts
    about ``overlap_chars`` before the end of the previous one, at the start of a word. Text
    without any whitespace cannot be overlapped cleanly and is cut without overlap.

    Args:
        text: Cleaned article text.
        max_chars: Largest allowed chunk length in characters.
        overlap_chars: Wanted overlap between neighbouring chunks, in characters.

    Returns:
        The chunks in reading order, numbered from 0 without gaps. Empty for blank text.

    Raises:
        ValidationError: If ``text`` is not a string, ``max_chars`` is not a positive integer,
            or ``overlap_chars`` is not an integer between 0 and ``max_chars - 1``.
    """
    if not isinstance(text, str):
        raise ValidationError("Text to split must be a string")
    if not _is_int(max_chars) or max_chars < 1:
        raise ValidationError("max_chars must be a positive integer")
    if not _is_int(overlap_chars) or overlap_chars < 0:
        raise ValidationError("overlap_chars must be a non-negative integer")
    if overlap_chars >= max_chars:
        raise ValidationError("overlap_chars must be smaller than max_chars")

    pieces: list[str] = []
    length = len(text)
    start = 0
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # the cut must stay past the overlap, or the next chunk could not move forward
            end = _find_break(text, start + max(max_chars // 2, overlap_chars + 1), end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = _next_start(text, start, end, overlap_chars)
    return [ChunkDraft(chunk_index=index, text=piece) for index, piece in enumerate(pieces)]
