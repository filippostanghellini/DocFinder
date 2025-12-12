"""Tests for PDF loading and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from docfinder.ingestion.pdf_loader import build_chunks, get_pdf_metadata, iter_text_parts
from docfinder.models import ChunkRecord


class TestIterTextParts:
    """Test iter_text_parts function."""

    @patch("docfinder.ingestion.pdf_loader.fitz")
    def test_iter_text_parts_simple(self, mock_fitz: MagicMock, tmp_path: Path) -> None:
        """Should extract text from PDF page by page."""
        # Setup mock
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text"

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz.open.return_value = mock_doc

        # Test
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        parts = list(iter_text_parts(pdf_path))

        assert len(parts) == 1
        assert "Page 1 text" in parts[0]

    @patch("docfinder.ingestion.pdf_loader.fitz")
    def test_iter_text_parts_multiple_pages(self, mock_fitz: MagicMock, tmp_path: Path) -> None:
        """Should extract text from multiple pages."""
        # Setup mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2"
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3"

        pages = [mock_page1, mock_page2, mock_page3]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

        mock_fitz.open.return_value = mock_doc

        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"dummy")

        parts = list(iter_text_parts(pdf_path))

        assert len(parts) == 3
        assert "Page 1" in parts[0]
        assert "Page 2" in parts[1]
        assert "Page 3" in parts[2]

    @patch("docfinder.ingestion.pdf_loader.fitz")
    @patch("docfinder.ingestion.pdf_loader.LOGGER")
    def test_iter_text_parts_page_error(
        self, mock_logger: MagicMock, mock_fitz: MagicMock, tmp_path: Path
    ) -> None:
        """Should log warning and continue on page extraction error."""
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1"

        mock_page2 = MagicMock()
        mock_page2.get_text.side_effect = Exception("Extraction failed")

        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Page 3"

        pages = [mock_page1, mock_page2, mock_page3]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

        mock_fitz.open.return_value = mock_doc

        pdf_path = tmp_path / "error.pdf"
        pdf_path.write_bytes(b"dummy")

        parts = list(iter_text_parts(pdf_path))

        # Should extract text from pages that work
        assert len(parts) == 2
        assert "Page 1" in parts[0]
        assert "Page 3" in parts[1]
        # Should have logged warning
        assert mock_logger.warning.called

    @patch("docfinder.ingestion.pdf_loader.fitz")
    @patch("docfinder.ingestion.pdf_loader.LOGGER")
    def test_iter_text_parts_open_error(
        self, mock_logger: MagicMock, mock_fitz: MagicMock, tmp_path: Path
    ) -> None:
        """Should log error and return empty on file open failure."""
        mock_fitz.open.side_effect = Exception("Cannot open file")

        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"dummy")

        parts = list(iter_text_parts(pdf_path))

        assert len(parts) == 0
        assert mock_logger.error.called


class TestGetPdfMetadata:
    """Test get_pdf_metadata function."""

    @patch("docfinder.ingestion.pdf_loader.fitz")
    def test_get_metadata_simple(self, mock_fitz: MagicMock, tmp_path: Path) -> None:
        """Should extract metadata from PDF."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.metadata = {"title": "Test PDF"}

        mock_fitz.open.return_value = mock_doc

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        metadata = get_pdf_metadata(pdf_path)

        assert metadata["title"] == "Test PDF"
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.fitz")
    def test_get_metadata_none(self, mock_fitz: MagicMock, tmp_path: Path) -> None:
        """Should handle PDF without metadata."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.metadata = None

        mock_fitz.open.return_value = mock_doc

        pdf_path = tmp_path / "no_meta.pdf"
        pdf_path.write_bytes(b"dummy")

        metadata = get_pdf_metadata(pdf_path)

        assert metadata["title"] == "no_meta"
        assert metadata["page_count"] == "1"

    @patch("docfinder.ingestion.pdf_loader.fitz")
    def test_get_metadata_empty_title(self, mock_fitz: MagicMock, tmp_path: Path) -> None:
        """Should use filename when title is empty."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.metadata = {"title": ""}

        mock_fitz.open.return_value = mock_doc

        pdf_path = tmp_path / "empty_title.pdf"
        pdf_path.write_bytes(b"dummy")

        metadata = get_pdf_metadata(pdf_path)

        assert metadata["title"] == "empty_title"
        assert metadata["page_count"] == "5"


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
