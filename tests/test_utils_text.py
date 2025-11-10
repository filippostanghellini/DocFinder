"""Tests for text utility functions."""

from __future__ import annotations

import pytest

from docfinder.utils.text import chunk_text, normalize_whitespace


class TestChunkText:
    """Test chunk_text function."""

    def test_chunk_short_text(self) -> None:
        """Should return single chunk for short text."""
        text = "Short text"
        chunks = list(chunk_text(text, max_chars=100, overlap=10))
        
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_long_text(self) -> None:
        """Should split long text into multiple chunks."""
        text = "a" * 500
        chunks = list(chunk_text(text, max_chars=100, overlap=20))
        
        assert len(chunks) > 1
        # Each chunk should be <= max_chars
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_chunk_overlap(self) -> None:
        """Should create overlapping chunks."""
        text = "0123456789" * 20  # 200 chars
        chunks = list(chunk_text(text, max_chars=100, overlap=20))
        
        # Should have overlap between consecutive chunks
        assert len(chunks) >= 2
        # Check that there's overlap
        if len(chunks) >= 2:
            # Last 20 chars of first chunk should overlap with second
            assert chunks[0][-20:] == chunks[1][:20]

    def test_chunk_empty_text(self) -> None:
        """Should handle empty text."""
        chunks = list(chunk_text("", max_chars=100, overlap=10))
        assert len(chunks) == 0

    def test_chunk_custom_sizes(self) -> None:
        """Should respect custom max_chars and overlap."""
        text = "x" * 1000
        chunks = list(chunk_text(text, max_chars=300, overlap=50))
        
        for chunk in chunks:
            assert len(chunk) <= 300


class TestNormalizeWhitespace:
    """Test normalize_whitespace function."""

    def test_normalize_simple(self) -> None:
        """Should join and strip lines."""
        lines = ["  Line 1  ", "  Line 2  ", "  Line 3  "]
        result = normalize_whitespace(lines)
        
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_empty_lines(self) -> None:
        """Should skip empty lines."""
        lines = ["Line 1", "", "  ", "Line 2", "\n", "Line 3"]
        result = normalize_whitespace(lines)
        
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_all_empty(self) -> None:
        """Should return empty string for all empty lines."""
        lines = ["", "  ", "\n", "\t"]
        result = normalize_whitespace(lines)
        
        assert result == ""

    def test_normalize_single_line(self) -> None:
        """Should handle single line."""
        lines = ["  Single line  "]
        result = normalize_whitespace(lines)
        
        assert result == "Single line"

    def test_normalize_preserves_content(self) -> None:
        """Should preserve content while removing extra whitespace."""
        lines = [
            "  Hello   World  ",
            "  This is a test  ",
            "",
            "  Final line  ",
        ]
        result = normalize_whitespace(lines)
        
        # Should preserve internal spaces but trim edges
        assert "Hello   World" in result
        assert "This is a test" in result
        assert "Final line" in result
