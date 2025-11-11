"""Tests for PDF loading and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from docfinder.ingestion.pdf_loader import build_chunks, extract_text
from docfinder.models import ChunkRecord


class TestExtractText:
    """Test extract_text function."""

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_extract_text_simple(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should extract text from PDF."""
        # Setup mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "Test PDF"

        mock_reader_class.return_value = mock_reader

        # Test
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        text, metadata = extract_text(pdf_path)

        assert "Page 1 text" in text
        assert metadata["title"] == "Test PDF"
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_extract_text_multiple_pages(
        self, mock_reader_class: MagicMock, tmp_path: Path
    ) -> None:
        """Should extract text from multiple pages."""
        # Setup mock
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2"
        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "Page 3"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "Multi-page PDF"

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"dummy")

        text, metadata = extract_text(pdf_path)

        assert "Page 1" in text
        assert "Page 2" in text
        assert "Page 3" in text
        assert metadata["page_count"] == "3"

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_extract_text_no_metadata(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should handle PDF without metadata."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = None

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "no_meta.pdf"
        pdf_path.write_bytes(b"dummy")

        text, metadata = extract_text(pdf_path)

        # Should use filename as title when no metadata
        assert metadata["title"] == "no_meta"
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_extract_text_empty_page(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should handle pages with no text."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None  # Empty page

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "Empty"

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"dummy")

        text, metadata = extract_text(pdf_path)

        # Should handle empty text gracefully
        assert isinstance(text, str)
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    @patch("docfinder.ingestion.pdf_loader.LOGGER")
    def test_extract_text_page_error(
        self, mock_logger: MagicMock, mock_reader_class: MagicMock, tmp_path: Path
    ) -> None:
        """Should log warning and continue on page extraction error."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1"

        mock_page2 = MagicMock()
        mock_page2.extract_text.side_effect = Exception("Extraction failed")

        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "Page 3"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "PDF with error"

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "error.pdf"
        pdf_path.write_bytes(b"dummy")

        text, metadata = extract_text(pdf_path)

        # Should extract text from pages that work
        assert "Page 1" in text
        assert "Page 3" in text
        # Should have logged warning
        assert mock_logger.warning.called


class TestBuildChunks:
    """Test build_chunks function."""

    @patch("docfinder.ingestion.pdf_loader.extract_text")
    def test_build_chunks_simple(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Should build chunks from PDF text."""
        mock_extract.return_value = (
            "This is a test document with some content.",
            {"title": "Test", "page_count": "1"},
        )

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path, max_chars=50, overlap=10))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkRecord) for chunk in chunks)
        assert chunks[0].document_path == pdf_path
        assert chunks[0].index == 0
        assert chunks[0].metadata["title"] == "Test"

    @patch("docfinder.ingestion.pdf_loader.extract_text")
    def test_build_chunks_multiple(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Should create multiple chunks for long text."""
        long_text = "x" * 500
        mock_extract.return_value = (long_text, {"title": "Long", "page_count": "1"})

        pdf_path = tmp_path / "long.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path, max_chars=100, overlap=20))

        # Should create multiple chunks
        assert len(chunks) > 1
        # Chunks should be indexed sequentially
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    @patch("docfinder.ingestion.pdf_loader.extract_text")
    def test_build_chunks_metadata(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Should include metadata in chunks."""
        mock_extract.return_value = (
            "Content",
            {"title": "My Document", "page_count": "5"},
        )

        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path))

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.metadata["title"] == "My Document"
        assert chunk.metadata["page_span"] == "5"

    @patch("docfinder.ingestion.pdf_loader.extract_text")
    def test_build_chunks_custom_size(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Should respect custom chunk size and overlap."""
        text = "a" * 1000
        mock_extract.return_value = (text, {"title": "Test", "page_count": "1"})

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path, max_chars=300, overlap=50))

        # All chunks should respect max_chars
        for chunk in chunks:
            assert len(chunk.text) <= 300

    @patch("docfinder.ingestion.pdf_loader.extract_text")
    def test_build_chunks_empty_text(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Should handle PDF with no extractable text."""
        mock_extract.return_value = ("", {"title": "Empty", "page_count": "1"})

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path))

        # Should return no chunks for empty text
        assert len(chunks) == 0
