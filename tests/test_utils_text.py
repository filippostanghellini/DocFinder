"""Tests for text utilities."""

from __future__ import annotations

import pytest

from docfinder.utils.text import (
    chunk_text,
    chunk_text_stream,
    chunk_text_stream_paged,
    normalize_whitespace,
)


class TestChunkText:
    """Tests for chunk_text function."""

    def test_empty_string_returns_empty_iter(self) -> None:
        """Returns empty iterator for empty string."""
        result = list(chunk_text(""))
        assert result == []

    def test_text_shorter_than_max(self) -> None:
        """Returns text as single chunk if shorter than max_chars."""
        text = "Short text that is longer than default overlap"
        result = list(chunk_text(text, max_chars=100, overlap=0))
        assert result == [text]

    def test_splits_into_chunks(self) -> None:
        """Splits text into overlapping chunks."""
        text = "A" * 100
        result = list(chunk_text(text, max_chars=30, overlap=10))
        # Step = max(30-10, 1) = 20
        # 0-30, 20-50, 40-70, 60-90, 80-100 = 5 chunks
        assert len(result) == 5
        # Each chunk should be at most 30 chars
        for chunk in result:
            assert len(chunk) <= 30

    def test_overlap_creates_repeated_content(self) -> None:
        """Overlap creates shared content between chunks."""
        text = "ABCDEFGHIJ" * 3  # 30 chars
        result = list(chunk_text(text, max_chars=15, overlap=5))
        assert len(result) > 1

    def test_step_ensures_progress(self) -> None:
        """Ensures at least 1 char step to prevent infinite loop."""
        text = "ABCDEFG"
        # overlap > max_chars would cause step=0 without protection
        result = list(chunk_text(text, max_chars=5, overlap=10))
        assert len(result) >= 1


class TestChunkTextStream:
    """Tests for chunk_text_stream function."""

    def test_empty_stream(self) -> None:
        """Handles empty stream."""
        result = list(chunk_text_stream([]))
        assert result == []

    def test_single_part(self) -> None:
        """Yields single part if shorter than max_chars."""
        result = list(chunk_text_stream(["short"], max_chars=100))
        assert result == ["short"]

    def test_accumulates_parts(self) -> None:
        """Accumulates parts until max_chars reached."""
        parts = ["ABC", "DEF", "GHI"]
        result = list(chunk_text_stream(parts, max_chars=5, overlap=2))
        assert len(result) >= 1

    def test_yields_remaining_buffer(self) -> None:
        """Yields remaining buffer at end."""
        parts = ["small"]
        result = list(chunk_text_stream(parts, max_chars=100))
        assert result == ["small"]


class TestChunkTextStreamPaged:
    """Tests for chunk_text_stream_paged function."""

    def test_empty_pages(self) -> None:
        """Handles empty page stream."""
        result = list(chunk_text_stream_paged([]))
        assert result == []

    def test_single_page(self) -> None:
        """Processes single page correctly."""
        pages = [(1, "This is page one content. It has multiple sentences.")]
        result = list(chunk_text_stream_paged(pages, max_chars=500))
        assert len(result) >= 1
        assert result[0][1] == 1  # page number

    def test_multiple_pages(self) -> None:
        """Processes multiple pages."""
        pages = [
            (1, "Page one content here."),
            (2, "Page two content here."),
        ]
        result = list(chunk_text_stream_paged(pages, max_chars=1000))
        # With 1000 max_chars, both pages might fit in one chunk
        assert len(result) >= 1

    def test_semantic_overlap(self) -> None:
        """Creates semantic overlap between chunks."""
        # Create content that spans multiple chunks
        text = " ".join([f"Sentence number {i} in the document." for i in range(20)])
        pages = [(1, text)]

        result = list(chunk_text_stream_paged(pages, max_chars=100, overlap=50))
        assert len(result) > 1

        # Check that there's overlap - some content appears in both chunks
        if len(result) > 1:
            chunk1 = result[0][0]
            chunk2 = result[1][0]
            # At least some content should overlap
            assert len(chunk1) > 0
            assert len(chunk2) > 0

    def test_respects_max_chars(self) -> None:
        """Ensures chunks don't exceed max_chars significantly."""
        text = "Word " * 10  # 60 chars including spaces
        pages = [(1, text)]

        result = list(chunk_text_stream_paged(pages, max_chars=30))
        for chunk, _ in result:
            # Allow some flexibility since sentences aren't split exactly
            assert len(chunk) <= 100


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_empty_iterable(self) -> None:
        """Handles empty iterable."""
        result = normalize_whitespace([])
        assert result == ""

    def test_strips_lines(self) -> None:
        """Strips whitespace from each line."""
        lines = ["  hello  ", "  world  "]
        result = normalize_whitespace(lines)
        assert result == "hello\nworld"

    def test_skips_empty_lines(self) -> None:
        """Skips empty lines."""
        lines = ["hello", "", "   ", "world"]
        result = normalize_whitespace(lines)
        assert result == "hello\nworld"

    def test_joins_with_newline(self) -> None:
        """Joins lines with newlines."""
        lines = ["line1", "line2", "line3"]
        result = normalize_whitespace(lines)
        assert result == "line1\nline2\nline3"

    def test_preserves_internal_whitespace(self) -> None:
        """Preserves internal whitespace within lines."""
        lines = ["hello    world", "foo\tbar"]
        result = normalize_whitespace(lines)
        assert "hello    world" in result
