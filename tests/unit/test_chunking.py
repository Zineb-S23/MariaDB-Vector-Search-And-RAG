"""Unit tests for deterministic text chunking."""

from __future__ import annotations

import itertools
from dataclasses import FrozenInstanceError

import pytest

from wikirag.errors import ValidationError
from wikirag.ingestion.chunking import ChunkDraft, split_into_chunks

# 200 unique words of 4 characters each, separated by single spaces
WORDS = " ".join(f"w{i:03d}" for i in range(200))


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_blank_text_gives_no_chunks(text: str) -> None:
    assert split_into_chunks(text) == []


def test_short_text_is_one_chunk() -> None:
    assert split_into_chunks("Hello world.") == [ChunkDraft(chunk_index=0, text="Hello world.")]


def test_text_of_exactly_max_chars_is_one_chunk() -> None:
    chunks = split_into_chunks("a" * 100, max_chars=100, overlap_chars=10)
    assert [chunk.text for chunk in chunks] == ["a" * 100]


def test_prefers_a_paragraph_break() -> None:
    text = "A" * 60 + "\n\n" + "B" * 60
    chunks = split_into_chunks(text, max_chars=100, overlap_chars=0)
    assert [chunk.text for chunk in chunks] == ["A" * 60, "B" * 60]


def test_cuts_after_a_sentence_end() -> None:
    text = "A" * 60 + ". " + "B" * 70
    chunks = split_into_chunks(text, max_chars=100, overlap_chars=0)
    assert [chunk.text for chunk in chunks] == ["A" * 60 + ".", "B" * 70]


def test_text_without_spaces_is_cut_hard_without_overlap() -> None:
    text = "x" * 250
    chunks = split_into_chunks(text, max_chars=100, overlap_chars=10)
    assert [len(chunk.text) for chunk in chunks] == [100, 100, 50]
    assert "".join(chunk.text for chunk in chunks) == text


def test_without_overlap_the_chunks_cover_the_text_exactly_once() -> None:
    chunks = split_into_chunks(WORDS, max_chars=100, overlap_chars=0)
    assert len(chunks) > 1
    assert " ".join(chunk.text for chunk in chunks).split() == WORDS.split()


def test_neighbouring_chunks_overlap_by_whole_words() -> None:
    chunks = split_into_chunks(WORDS, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    for previous, current in itertools.pairwise(chunks):
        shared = [word for word in current.text.split() if word in previous.text.split()]
        assert shared
        assert len(" ".join(shared)) <= 20


@pytest.mark.parametrize(("max_chars", "overlap_chars"), [(100, 0), (100, 20), (50, 49), (30, 10)])
def test_chunks_respect_the_size_limit_and_numbering(max_chars: int, overlap_chars: int) -> None:
    chunks = split_into_chunks(WORDS, max_chars=max_chars, overlap_chars=overlap_chars)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text and len(chunk.text) <= max_chars for chunk in chunks)
    assert all(chunk.text == chunk.text.strip() for chunk in chunks)
    assert all(chunk.section_title is None for chunk in chunks)


def test_default_settings_keep_the_whole_text() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(30)).strip()
    chunks = split_into_chunks(text)
    assert all(len(chunk.text) <= 1000 for chunk in chunks)
    assert chunks[0].text.startswith("Paragraph 0.")
    assert chunks[-1].text.endswith("word")


def test_same_input_gives_the_same_chunks() -> None:
    first = split_into_chunks(WORDS, max_chars=100, overlap_chars=20)
    assert split_into_chunks(WORDS, max_chars=100, overlap_chars=20) == first


@pytest.mark.parametrize("max_chars", [0, -5, 1.5, True, "100"])
def test_invalid_max_chars_raises(max_chars: object) -> None:
    with pytest.raises(ValidationError, match="max_chars"):
        split_into_chunks("text", max_chars=max_chars, overlap_chars=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("overlap_chars", [-1, 1.5, True, "10"])
def test_invalid_overlap_raises(overlap_chars: object) -> None:
    with pytest.raises(ValidationError, match="overlap_chars"):
        split_into_chunks("text", max_chars=100, overlap_chars=overlap_chars)  # type: ignore[arg-type]


@pytest.mark.parametrize("overlap_chars", [100, 150])
def test_overlap_not_smaller_than_max_raises(overlap_chars: int) -> None:
    with pytest.raises(ValidationError, match="smaller"):
        split_into_chunks("text", max_chars=100, overlap_chars=overlap_chars)


@pytest.mark.parametrize("value", [None, 5, b"bytes", ["text"]])
def test_non_string_raises(value: object) -> None:
    with pytest.raises(ValidationError, match="string"):
        split_into_chunks(value)  # type: ignore[arg-type]


def test_chunk_draft_cannot_be_changed() -> None:
    chunk = ChunkDraft(chunk_index=0, text="abc")
    with pytest.raises(FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]
