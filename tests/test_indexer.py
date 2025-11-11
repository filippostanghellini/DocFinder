"""Tests for Indexer."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from docfinder.index.indexer import Indexer, IndexStats
from docfinder.models import ChunkRecord


class TestIndexStats:
    """Test IndexStats tracking."""

    def test_init_defaults(self):
        """Test default initialization."""
        stats = IndexStats()
        assert stats.inserted == 0
        assert stats.updated == 0
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.processed_files == []

    def test_increment_inserted(self):
        """Test incrementing inserted count."""
        stats = IndexStats()
        path = Path("/tmp/test.pdf")

        stats.increment("inserted", path)

        assert stats.inserted == 1
        assert stats.updated == 0
        assert stats.skipped == 0
        assert stats.failed == 0
        assert path in stats.processed_files

    def test_increment_updated(self):
        """Test incrementing updated count."""
        stats = IndexStats()
        path = Path("/tmp/test.pdf")

        stats.increment("updated", path)

        assert stats.inserted == 0
        assert stats.updated == 1
        assert stats.skipped == 0
        assert stats.failed == 0
        assert path in stats.processed_files

    def test_increment_skipped(self):
        """Test incrementing skipped count."""
        stats = IndexStats()
        path = Path("/tmp/test.pdf")

        stats.increment("skipped", path)

        assert stats.inserted == 0
        assert stats.updated == 0
        assert stats.skipped == 1
        assert stats.failed == 0
        assert path in stats.processed_files

    def test_increment_failed(self):
        """Test incrementing failed count for unknown status."""
        stats = IndexStats()
        path = Path("/tmp/test.pdf")

        stats.increment("unknown_status", path)

        assert stats.inserted == 0
        assert stats.updated == 0
        assert stats.skipped == 0
        assert stats.failed == 1
        assert path in stats.processed_files

    def test_multiple_increments(self):
        """Test tracking multiple files."""
        stats = IndexStats()

        stats.increment("inserted", Path("/tmp/a.pdf"))
        stats.increment("updated", Path("/tmp/b.pdf"))
        stats.increment("skipped", Path("/tmp/c.pdf"))
        stats.increment("failed", Path("/tmp/d.pdf"))

        assert stats.inserted == 1
        assert stats.updated == 1
        assert stats.skipped == 1
        assert stats.failed == 1
        assert len(stats.processed_files) == 4


class TestIndexer:
    """Test Indexer document pipeline."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock EmbeddingModel."""
        embedder = Mock()
        embedder.embed.return_value = np.random.rand(2, 384).astype("float32")
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock SQLiteVectorStore."""
        store = Mock()
        store.upsert_document.return_value = "inserted"
        return store

    @pytest.fixture
    def indexer(self, mock_embedder, mock_store):
        """Create Indexer instance."""
        return Indexer(mock_embedder, mock_store, chunk_chars=1200, overlap=200)

    def test_init(self, mock_embedder, mock_store):
        """Test Indexer initialization."""
        indexer = Indexer(mock_embedder, mock_store, chunk_chars=1000, overlap=100)

        assert indexer.embedder is mock_embedder
        assert indexer.store is mock_store
        assert indexer.chunk_chars == 1000
        assert indexer.overlap == 100

    def test_init_default_params(self, mock_embedder, mock_store):
        """Test default chunk parameters."""
        indexer = Indexer(mock_embedder, mock_store)

        assert indexer.chunk_chars == 1200
        assert indexer.overlap == 200

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_single_file(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing a single file."""
        # Setup
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(
                document_path=pdf_path, index=0, text="Chunk 1", metadata={"title": "Test Doc"}
            ),
            ChunkRecord(document_path=pdf_path, index=1, text="Chunk 2", metadata={}),
        ]
        mock_sha256.return_value = "abc123"

        # Execute
        stats = indexer.index([pdf_path])

        # Verify
        assert stats.inserted == 1
        assert stats.updated == 0
        assert stats.skipped == 0
        assert stats.failed == 0
        assert pdf_path in stats.processed_files

        # Verify calls
        mock_build_chunks.assert_called_once_with(pdf_path, max_chars=1200, overlap=200)
        mock_sha256.assert_called_once_with(pdf_path)
        indexer.embedder.embed.assert_called_once_with(["Chunk 1", "Chunk 2"])
        indexer.store.upsert_document.assert_called_once()

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_updated_file(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing a file that gets updated."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=pdf_path, index=0, text="Updated", metadata={})
        ]
        mock_sha256.return_value = "new_hash"
        indexer.store.upsert_document.return_value = "updated"

        stats = indexer.index([pdf_path])

        assert stats.updated == 1
        assert stats.inserted == 0

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_skipped_file(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing a file that gets skipped."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=pdf_path, index=0, text="Same", metadata={})
        ]
        mock_sha256.return_value = "same_hash"
        indexer.store.upsert_document.return_value = "skipped"

        stats = indexer.index([pdf_path])

        assert stats.skipped == 1
        assert stats.inserted == 0

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    def test_index_empty_document(self, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path):
        """Test indexing a document with no extractable text."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = []  # No chunks

        stats = indexer.index([pdf_path])

        assert stats.skipped == 1
        assert stats.inserted == 0
        # Should not call embedder or store
        indexer.embedder.embed.assert_not_called()
        indexer.store.upsert_document.assert_not_called()

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    def test_index_exception_handling(self, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path):
        """Test that exceptions are caught and logged."""
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.side_effect = Exception("PDF parsing error")

        stats = indexer.index([pdf_path])

        assert stats.failed == 1
        assert stats.inserted == 0
        assert pdf_path in stats.processed_files

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_multiple_files(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing multiple files."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_text("test1")
        pdf2.write_text("test2")

        mock_iter_pdfs.return_value = [pdf1, pdf2]
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=Path("/tmp/dummy.pdf"), index=0, text="Text", metadata={})
        ]
        mock_sha256.side_effect = ["hash1", "hash2"]
        indexer.store.upsert_document.side_effect = ["inserted", "inserted"]

        stats = indexer.index([tmp_path])

        assert stats.inserted == 2
        assert len(stats.processed_files) == 2

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_mixed_results(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing with mixed success/skip/fail."""
        pdf1 = tmp_path / "new.pdf"
        pdf2 = tmp_path / "same.pdf"
        pdf3 = tmp_path / "broken.pdf"
        pdf1.write_text("new")
        pdf2.write_text("same")
        pdf3.write_text("broken")

        mock_iter_pdfs.return_value = [pdf1, pdf2, pdf3]

        # pdf1: new insert, pdf2: skipped, pdf3: exception
        def build_chunks_side_effect(path, **kwargs):
            if path == pdf3:
                raise Exception("Error")
            return [ChunkRecord(document_path=path, index=0, text="Text", metadata={})]

        mock_build_chunks.side_effect = build_chunks_side_effect
        mock_sha256.side_effect = ["hash1", "hash2"]
        indexer.store.upsert_document.side_effect = ["inserted", "skipped"]

        stats = indexer.index([tmp_path])

        assert stats.inserted == 1
        assert stats.skipped == 1
        assert stats.failed == 1
        assert len(stats.processed_files) == 3

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_uses_title_from_metadata(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that document title is extracted from first chunk metadata."""
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(
                document_path=pdf_path, index=0, text="Text", metadata={"title": "Custom Title"}
            ),
            ChunkRecord(document_path=pdf_path, index=1, text="More", metadata={}),
        ]
        mock_sha256.return_value = "abc123"

        indexer.index([pdf_path])

        # Verify document metadata
        call_args = indexer.store.upsert_document.call_args
        document = call_args[0][0]
        assert document.title == "Custom Title"

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_uses_filename_when_no_title(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that filename is used when no title in metadata."""
        pdf_path = tmp_path / "my_document.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=pdf_path, index=0, text="Text", metadata={})  # No title
        ]
        mock_sha256.return_value = "abc123"

        indexer.index([pdf_path])

        # Verify uses filename stem
        call_args = indexer.store.upsert_document.call_args
        document = call_args[0][0]
        assert document.title == "my_document"

    @patch("docfinder.index.indexer.iter_pdf_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_preserves_file_stats(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that file stats (mtime, size) are captured."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test content")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=pdf_path, index=0, text="Text", metadata={})
        ]
        mock_sha256.return_value = "abc123"

        indexer.index([pdf_path])

        # Verify document metadata includes stats
        call_args = indexer.store.upsert_document.call_args
        document = call_args[0][0]
        assert document.mtime == pdf_path.stat().st_mtime
        assert document.size == pdf_path.stat().st_size
