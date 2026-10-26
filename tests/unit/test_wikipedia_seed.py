"""Unit tests for loading the Wikipedia seed title list."""

from __future__ import annotations

from pathlib import Path

import pytest

from wikirag.errors import ValidationError
from wikirag.ingestion.wikipedia import load_seed_titles

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_seed(tmp_path: Path, content: str) -> Path:
    """Write seed text to a temporary file and return its path."""
    path = tmp_path / "seed.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_pairs_in_file_order(tmp_path: Path) -> None:
    path = write_seed(tmp_path, "topic,title\nphysics,Black hole\nbiology,DNA\n")
    assert load_seed_titles(path) == [("physics", "Black hole"), ("biology", "DNA")]


def test_strips_whitespace_and_normalizes_underscores(tmp_path: Path) -> None:
    path = write_seed(tmp_path, "topic,title\n physics , Black_hole  \n")
    assert load_seed_titles(path) == [("physics", "Black hole")]


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = write_seed(tmp_path, "topic,title\n\nphysics,Black hole\n\n")
    assert load_seed_titles(path) == [("physics", "Black hole")]


def test_accepts_byte_order_mark_and_quoted_commas(tmp_path: Path) -> None:
    path = tmp_path / "seed.csv"
    path.write_bytes('\ufefftopic,title\nbiology,"Cat, domestic"\n'.encode())
    assert load_seed_titles(path) == [("biology", "Cat, domestic")]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        load_seed_titles(tmp_path / "missing.csv")


@pytest.mark.parametrize("content", ["", "title,topic\nBlack hole,physics\n", "name\nDNA\n"])
def test_wrong_header_raises(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValidationError, match="header"):
        load_seed_titles(write_seed(tmp_path, content))


def test_header_only_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no seed rows"):
        load_seed_titles(write_seed(tmp_path, "topic,title\n"))


@pytest.mark.parametrize("row", ["physics,   ", " ,Black hole", ",Black hole"])
def test_empty_topic_or_title_raises(tmp_path: Path, row: str) -> None:
    with pytest.raises(ValidationError, match="Line 2"):
        load_seed_titles(write_seed(tmp_path, f"topic,title\n{row}\n"))


@pytest.mark.parametrize("row", ["physics", "physics,Black hole,extra"])
def test_wrong_column_count_raises(tmp_path: Path, row: str) -> None:
    with pytest.raises(ValidationError, match="Line 2"):
        load_seed_titles(write_seed(tmp_path, f"topic,title\n{row}\n"))


def test_duplicate_title_raises_even_with_different_spelling(tmp_path: Path) -> None:
    path = write_seed(tmp_path, "topic,title\nphysics,Black hole\nastronomy,black_hole\n")
    with pytest.raises(ValidationError, match="duplicate"):
        load_seed_titles(path)


def test_invalid_utf8_raises(tmp_path: Path) -> None:
    path = tmp_path / "seed.csv"
    path.write_bytes(b"topic,title\nphysics,\xff\xfe\n")
    with pytest.raises(ValidationError, match="UTF-8"):
        load_seed_titles(path)


def test_committed_seed_file_is_valid() -> None:
    seeds = load_seed_titles(REPO_ROOT / "data" / "seed_titles.csv")
    assert seeds
    assert all(topic and title for topic, title in seeds)
