"""Tests for PDF loading and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from docfinder.ingestion.pdf_loader import build_chunks, get_pdf_metadata, iter_text_parts
from docfinder.models import ChunkRecord


class TestIterTextParts:
    """Test iter_text_parts function."""

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_iter_text_parts_simple(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should extract text from PDF page by page."""
        # Setup mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_reader_class.return_value = mock_reader

        # Test
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        parts = list(iter_text_parts(pdf_path))

        assert len(parts) == 1
        assert "Page 1 text" in parts[0]

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_iter_text_parts_multiple_pages(
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

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"dummy")

        parts = list(iter_text_parts(pdf_path))

        assert len(parts) == 3
        assert "Page 1" in parts[0]
        assert "Page 2" in parts[1]
        assert "Page 3" in parts[2]

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    @patch("docfinder.ingestion.pdf_loader.LOGGER")
    def test_iter_text_parts_page_error(
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

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "error.pdf"
        pdf_path.write_bytes(b"dummy")

        parts = list(iter_text_parts(pdf_path))

        # Should extract text from pages that work
        assert len(parts) == 2
        assert "Page 1" in parts[0]
        assert "Page 3" in parts[1]
        # Should have logged warning
        assert mock_logger.warning.called


class TestGetPdfMetadata:
    """Test get_pdf_metadata function."""

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_get_metadata_simple(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should extract metadata from PDF."""
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]
        mock_reader.metadata = MagicMock()
        mock_reader.metadata.title = "Test PDF"

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        metadata = get_pdf_metadata(pdf_path)

        assert metadata["title"] == "Test PDF"
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.PdfReader")
    def test_get_metadata_none(self, mock_reader_class: MagicMock, tmp_path: Path) -> None:
        """Should handle PDF without metadata."""
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]
        mock_reader.metadata = None

        mock_reader_class.return_value = mock_reader

        pdf_path = tmp_path / "no_meta.pdf"
        pdf_path.write_bytes(b"dummy")

        metadata = get_pdf_metadata(pdf_path)

        assert metadata["title"] == "no_meta"
        assert metadata["page_count"] == "1"


class TestBuildChunks:
    """Test build_chunks function."""

    @patch("docfinder.ingestion.pdf_loader.iter_text_parts")
    @patch("docfinder.ingestion.pdf_loader.get_pdf_metadata")
    def test_build_chunks_simple(
        self, mock_meta: MagicMock, mock_iter: MagicMock, tmp_path: Path
    ) -> None:
        """Should build chunks from PDF text."""
        mock_meta.return_value = {"title": "Test", "page_count": "1"}
        mock_iter.return_value = iter(["This is a test document with some content."])

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path, max_chars=50, overlap=10))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkRecord) for chunk in chunks)
        assert chunks[0].document_path == pdf_path
        assert chunks[0].index == 0
        assert chunks[0].metadata["title"] == "Test"

    @patch("docfinder.ingestion.pdf_loader.iter_text_parts")
    @patch("docfinder.ingestion.pdf_loader.get_pdf_metadata")
    def test_build_chunks_multiple(
        self, mock_meta: MagicMock, mock_iter: MagicMock, tmp_path: Path
    ) -> None:
        """Should create multiple chunks for long text."""
        long_text = "x" * 500
        mock_meta.return_value = {"title": "Long", "page_count": "1"}
        mock_iter.return_value = iter([long_text])

        pdf_path = tmp_path / "long.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path, max_chars=100, overlap=20))

        # Should create multiple chunks
        assert len(chunks) > 1
        # Chunks should be indexed sequentially
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    @patch("docfinder.ingestion.pdf_loader.iter_text_parts")
    @patch("docfinder.ingestion.pdf_loader.get_pdf_metadata")
    def test_build_chunks_metadata(
        self, mock_meta: MagicMock, mock_iter: MagicMock, tmp_path: Path
    ) -> None:
        """Should include metadata in chunks."""
        mock_meta.return_value = {"title": "My Document", "page_count": "5"}
        mock_iter.return_value = iter(["Content"])

        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path))

        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.metadata["title"] == "My Document"
        assert chunk.metadata["page_span"] == "5"

    @patch("docfinder.ingestion.pdf_loader.iter_text_parts")
    @patch("docfinder.ingestion.pdf_loader.get_pdf_metadata")
    def test_build_chunks_empty_text(
        self, mock_meta: MagicMock, mock_iter: MagicMock, tmp_path: Path
    ) -> None:
        """Should handle PDF with no extractable text."""
        mock_meta.return_value = {"title": "Empty", "page_count": "1"}
        mock_iter.return_value = iter([])

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"dummy")

        chunks = list(build_chunks(pdf_path))

        # Should return no chunks for empty text
        assert len(chunks) == 0
