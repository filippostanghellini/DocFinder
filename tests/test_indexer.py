"""Tests for Indexer."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from docfinder.index.indexer import Indexer, IndexStats, _parse_document
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
        store.init_document.return_value = (1, "inserted")
        store.insert_chunks.return_value = None
        store.transaction.return_value.__enter__ = Mock()
        store.transaction.return_value.__exit__ = Mock()
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

    @patch("docfinder.index.indexer.iter_document_paths")
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
        # Must return an iterator
        mock_build_chunks.return_value = iter(
            [
                ChunkRecord(
                    document_path=pdf_path, index=0, text="Chunk 1", metadata={"title": "Test Doc"}
                ),
                ChunkRecord(document_path=pdf_path, index=1, text="Chunk 2", metadata={}),
            ]
        )
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
        # Should be called once for the batch of 2 (since batch size is 32)
        indexer.embedder.embed.assert_called_once()
        indexer.store.init_document.assert_called_once()
        indexer.store.insert_chunks.assert_called_once()

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_updated_file(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing a file that gets updated."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter(
            [ChunkRecord(document_path=pdf_path, index=0, text="Updated", metadata={})]
        )
        mock_sha256.return_value = "new_hash"
        indexer.store.init_document.return_value = (1, "updated")

        stats = indexer.index([pdf_path])

        assert stats.updated == 1
        assert stats.inserted == 0

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_skipped_file(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test indexing a file that gets skipped."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter(
            [ChunkRecord(document_path=pdf_path, index=0, text="Same", metadata={})]
        )
        mock_sha256.return_value = "same_hash"
        indexer.store.init_document.return_value = (-1, "skipped")

        stats = indexer.index([pdf_path])

        assert stats.skipped == 1
        assert stats.inserted == 0
        # Should not embed or insert chunks
        indexer.embedder.embed.assert_not_called()
        indexer.store.insert_chunks.assert_not_called()

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    def test_index_empty_document(self, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path):
        """Test indexing a document with no extractable text."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter([])  # Empty iterator

        stats = indexer.index([pdf_path])

        assert stats.skipped == 1
        assert stats.inserted == 0
        # Should not call embedder or store
        indexer.embedder.embed.assert_not_called()
        indexer.store.init_document.assert_not_called()

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    def test_index_exception_handling(self, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path):
        """Test that exceptions are caught and logged."""
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]

        # Exception when iterating
        def error_gen():
            raise Exception("PDF parsing error")
            yield  # unreachable

        mock_build_chunks.return_value = error_gen()

        stats = indexer.index([pdf_path])

        assert stats.failed == 1
        assert stats.inserted == 0
        assert pdf_path in stats.processed_files

    @patch("docfinder.index.indexer.iter_document_paths")
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

        # We need side_effect for build_chunks to return fresh iterators
        mock_build_chunks.side_effect = [
            iter([ChunkRecord(document_path=pdf1, index=0, text="Text1", metadata={})]),
            iter([ChunkRecord(document_path=pdf2, index=0, text="Text2", metadata={})]),
        ]

        mock_sha256.side_effect = ["hash1", "hash2"]
        indexer.store.init_document.side_effect = [(1, "inserted"), (2, "inserted")]

        stats = indexer.index([tmp_path])

        assert stats.inserted == 2
        assert len(stats.processed_files) == 2

    @patch("docfinder.index.indexer.iter_document_paths")
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
            return iter([ChunkRecord(document_path=path, index=0, text="Text", metadata={})])

        mock_build_chunks.side_effect = build_chunks_side_effect
        mock_sha256.side_effect = ["hash1", "hash2"]
        indexer.store.init_document.side_effect = [(1, "inserted"), (-1, "skipped")]

        stats = indexer.index([tmp_path])

        assert stats.inserted == 1
        assert stats.skipped == 1
        assert stats.failed == 1
        assert len(stats.processed_files) == 3

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_uses_title_from_metadata(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that document title is extracted from first chunk metadata."""
        pdf_path = tmp_path / "document.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter(
            [
                ChunkRecord(
                    document_path=pdf_path, index=0, text="Text", metadata={"title": "Custom Title"}
                ),
                ChunkRecord(document_path=pdf_path, index=1, text="More", metadata={}),
            ]
        )
        mock_sha256.return_value = "abc123"
        indexer.store.init_document.return_value = (1, "inserted")

        indexer.index([pdf_path])

        # Verify document metadata passed to init_document
        call_args = indexer.store.init_document.call_args
        document = call_args[0][0]
        assert document.title == "Custom Title"

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_uses_filename_when_no_title(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that filename is used when no title in metadata."""
        pdf_path = tmp_path / "my_document.pdf"
        pdf_path.write_text("test")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter(
            [
                ChunkRecord(document_path=pdf_path, index=0, text="Text", metadata={})  # No title
            ]
        )
        mock_sha256.return_value = "abc123"
        indexer.store.init_document.return_value = (1, "inserted")

        indexer.index([pdf_path])

        # Verify uses filename stem
        call_args = indexer.store.init_document.call_args
        document = call_args[0][0]
        assert document.title == "my_document"

    @patch("docfinder.index.indexer.iter_document_paths")
    @patch("docfinder.index.indexer.build_chunks")
    @patch("docfinder.index.indexer.compute_sha256")
    def test_index_preserves_file_stats(
        self, mock_sha256, mock_build_chunks, mock_iter_pdfs, indexer, tmp_path
    ):
        """Test that file stats (mtime, size) are captured."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test content")

        mock_iter_pdfs.return_value = [pdf_path]
        mock_build_chunks.return_value = iter(
            [ChunkRecord(document_path=pdf_path, index=0, text="Text", metadata={})]
        )
        mock_sha256.return_value = "abc123"
        indexer.store.init_document.return_value = (1, "inserted")

        indexer.index([pdf_path])

        # Verify document metadata includes stats
        call_args = indexer.store.init_document.call_args
        document = call_args[0][0]
        assert document.mtime == pdf_path.stat().st_mtime
        assert document.size == pdf_path.stat().st_size


class TestParseDocument:
    """Test the module-level _parse_document function."""

    @patch("docfinder.index.indexer.compute_sha256")
    @patch("docfinder.index.indexer.build_chunks")
    def test_parse_document_success(self, mock_build_chunks, mock_sha256, tmp_path):
        """Test successful document parsing."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("test content")
        mock_sha256.return_value = "abc123"
        mock_build_chunks.return_value = [
            ChunkRecord(document_path=pdf_path, index=0, text="Chunk 1", metadata={"title": "T"}),
            ChunkRecord(document_path=pdf_path, index=1, text="Chunk 2", metadata={}),
        ]

        result = _parse_document((str(pdf_path), 1200, 200))

        assert result["status"] == "ok"
        assert result["path"] == str(pdf_path)
        assert result["sha256"] == "abc123"
        assert result["title"] == "T"
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["text"] == "Chunk 1"
        assert result["chunks"][1]["index"] == 1

    @patch("docfinder.index.indexer.build_chunks")
    def test_parse_document_empty(self, mock_build_chunks, tmp_path):
        """Test parsing a document with no extractable text."""
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_text("empty")
        mock_build_chunks.return_value = []

        result = _parse_document((str(pdf_path), 1200, 200))

        assert result["status"] == "empty"
        assert result["path"] == str(pdf_path)

    @patch("docfinder.index.indexer.build_chunks")
    def test_parse_document_error(self, mock_build_chunks, tmp_path):
        """Test parsing a document that raises an exception."""
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_text("broken")
        mock_build_chunks.side_effect = Exception("Parse error")

        result = _parse_document((str(pdf_path), 1200, 200))

        assert result["status"] == "error"
        assert result["path"] == str(pdf_path)
        assert "Parse error" in result["error"]


class TestParallelIndexing:
    """Test parallel indexing path."""

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
        store.init_document.return_value = (1, "inserted")
        store.insert_chunks.return_value = None
        store.transaction.return_value.__enter__ = Mock()
        store.transaction.return_value.__exit__ = Mock()
        return store

    def test_should_parallelize_true_by_default(self, mock_embedder, mock_store):
        """Test that _should_parallelize returns True when no fixed batch size."""
        indexer = Indexer(mock_embedder, mock_store)
        assert indexer._should_parallelize() is True

    def test_should_parallelize_false_with_fixed_batch(self, mock_embedder, mock_store):
        """Test that _should_parallelize returns False with fixed embed_batch_size."""
        indexer = Indexer(mock_embedder, mock_store, embed_batch_size=32)
        assert indexer._should_parallelize() is False

    @patch("docfinder.index.indexer.iter_document_paths")
    def test_sequential_path_used_for_few_files(
        self, mock_iter_pdfs, mock_embedder, mock_store, tmp_path
    ):
        """Test that sequential path is used when < 4 files."""
        files = []
        for i in range(3):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        indexer = Indexer(mock_embedder, mock_store)

        with (
            patch.object(indexer, "_index_sequential") as mock_seq,
            patch.object(indexer, "_index_parallel") as mock_par,
        ):
            indexer.index([tmp_path])
            mock_seq.assert_called_once()
            mock_par.assert_not_called()

    @patch("docfinder.index.indexer.iter_document_paths")
    def test_parallel_path_used_for_many_files(
        self, mock_iter_pdfs, mock_embedder, mock_store, tmp_path
    ):
        """Test that parallel path is used when >= 4 files."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        indexer = Indexer(mock_embedder, mock_store)

        with (
            patch.object(indexer, "_index_sequential") as mock_seq,
            patch.object(indexer, "_index_parallel") as mock_par,
        ):
            indexer.index([tmp_path])
            mock_par.assert_called_once()
            mock_seq.assert_not_called()

    @patch("docfinder.index.indexer.iter_document_paths")
    def test_sequential_path_when_batch_size_set(
        self, mock_iter_pdfs, mock_embedder, mock_store, tmp_path
    ):
        """Test that sequential path is used when embed_batch_size is set, even with many files."""
        files = []
        for i in range(5):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        indexer = Indexer(mock_embedder, mock_store, embed_batch_size=32)

        with (
            patch.object(indexer, "_index_sequential") as mock_seq,
            patch.object(indexer, "_index_parallel") as mock_par,
        ):
            indexer.index([tmp_path])
            mock_seq.assert_called_once()
            mock_par.assert_not_called()

    @patch("docfinder.index.indexer.ProcessPoolExecutor")
    @patch("docfinder.index.indexer.iter_document_paths")
    def test_parallel_indexing_with_mocked_executor(
        self, mock_iter_pdfs, mock_executor_cls, mock_embedder, mock_store, tmp_path
    ):
        """Test parallel indexing end-to-end with mocked ProcessPoolExecutor."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        # Simulate parsed results from worker processes
        parsed_results = [
            {
                "path": str(files[0]),
                "status": "ok",
                "sha256": "h0",
                "mtime": 1000.0,
                "size": 100,
                "title": "Doc 0",
                "chunks": [{"index": 0, "text": "text0", "metadata": {"title": "Doc 0"}}],
            },
            {
                "path": str(files[1]),
                "status": "ok",
                "sha256": "h1",
                "mtime": 1001.0,
                "size": 101,
                "title": "Doc 1",
                "chunks": [{"index": 0, "text": "text1", "metadata": {"title": "Doc 1"}}],
            },
            {"path": str(files[2]), "status": "empty"},
            {"path": str(files[3]), "status": "error", "error": "bad pdf"},
        ]

        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = parsed_results
        mock_executor_cls.return_value = mock_executor

        mock_store.init_document.side_effect = [(1, "inserted"), (2, "inserted")]

        indexer = Indexer(mock_embedder, mock_store)
        stats = indexer.index([tmp_path])

        assert stats.inserted == 2
        assert stats.skipped == 1
        assert stats.failed == 1
        assert len(stats.processed_files) == 4
        assert mock_embedder.embed.call_count == 2
        assert mock_store.insert_chunks.call_count == 2

    @patch("docfinder.index.indexer.ProcessPoolExecutor")
    @patch("docfinder.index.indexer.iter_document_paths")
    def test_parallel_progress_callback(
        self, mock_iter_pdfs, mock_executor_cls, mock_embedder, mock_store, tmp_path
    ):
        """Test that progress callback fires during parallel indexing."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        parsed_results = [
            {
                "path": str(f),
                "status": "ok",
                "sha256": f"h{i}",
                "mtime": 1000.0,
                "size": 100,
                "title": f"Doc {i}",
                "chunks": [{"index": 0, "text": f"text{i}", "metadata": {}}],
            }
            for i, f in enumerate(files)
        ]

        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = parsed_results
        mock_executor_cls.return_value = mock_executor

        mock_store.init_document.side_effect = [(i, "inserted") for i in range(1, 5)]

        callback = Mock()
        indexer = Indexer(mock_embedder, mock_store, progress_callback=callback)
        indexer.index([tmp_path])

        # 4 per-file calls + 1 final completion call
        assert callback.call_count == 5
        # Final call should be (total, total, "")
        callback.assert_any_call(4, 4, "")

    @patch("docfinder.index.indexer.ProcessPoolExecutor")
    @patch("docfinder.index.indexer.iter_document_paths")
    def test_parallel_skipped_by_store(
        self, mock_iter_pdfs, mock_executor_cls, mock_embedder, mock_store, tmp_path
    ):
        """Test that store-level skip works in parallel path."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_text(f"test{i}")
            files.append(p)
        mock_iter_pdfs.return_value = files

        parsed_results = [
            {
                "path": str(f),
                "status": "ok",
                "sha256": f"h{i}",
                "mtime": 1000.0,
                "size": 100,
                "title": f"Doc {i}",
                "chunks": [{"index": 0, "text": f"text{i}", "metadata": {}}],
            }
            for i, f in enumerate(files)
        ]

        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=False)
        mock_executor.map.return_value = parsed_results
        mock_executor_cls.return_value = mock_executor

        # All documents already indexed (skipped)
        mock_store.init_document.return_value = (-1, "skipped")

        indexer = Indexer(mock_embedder, mock_store)
        stats = indexer.index([tmp_path])

        assert stats.skipped == 4
        assert stats.inserted == 0
        mock_embedder.embed.assert_not_called()
        mock_store.insert_chunks.assert_not_called()
