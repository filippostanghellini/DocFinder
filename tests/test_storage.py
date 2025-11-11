"""Tests for SQLiteVectorStore."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from docfinder.index.storage import SQLiteVectorStore
from docfinder.models import ChunkRecord, DocumentMetadata


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    store = SQLiteVectorStore(db_path, dimension=384)
    yield store
    store.close()


class TestSQLiteVectorStore:
    """Test SQLiteVectorStore initialization and schema."""

    def test_init_creates_database(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        store = SQLiteVectorStore(db_path, dimension=384)

        assert db_path.exists()
        assert store.db_path == db_path
        assert store.dimension == 384
        store.close()

    def test_schema_creation(self, temp_db):
        """Test that schema is properly created."""
        conn = temp_db.connection

        # Check documents table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        assert cursor.fetchone() is not None

        # Check chunks table
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        assert cursor.fetchone() is not None

        # Check index
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_document_id'"
        )
        assert cursor.fetchone() is not None

    def test_pragma_settings(self, temp_db):
        """Test that PRAGMA settings are applied."""
        conn = temp_db.connection

        # Check WAL mode
        cursor = conn.execute("PRAGMA journal_mode")
        result = cursor.fetchone()
        assert result[0].lower() == "wal"

        # Check synchronous mode
        cursor = conn.execute("PRAGMA synchronous")
        result = cursor.fetchone()
        assert result[0] == 1  # NORMAL mode

    def test_connection_property(self, temp_db):
        """Test connection property returns sqlite3.Connection."""
        assert isinstance(temp_db.connection, sqlite3.Connection)

    def test_close(self, tmp_path):
        """Test closing the database connection."""
        db_path = tmp_path / "close_test.db"
        store = SQLiteVectorStore(db_path, dimension=384)
        conn = store.connection

        store.close()

        # After closing, operations should fail
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestTransaction:
    """Test transaction context manager."""

    def test_commit_on_success(self, temp_db):
        """Test that transaction commits on success."""
        with temp_db.transaction() as conn:
            conn.execute(
                "INSERT INTO documents(path, title, sha256, mtime, size) VALUES (?, ?, ?, ?, ?)",
                ("/tmp/test.pdf", "Test", "abc123", 1234567890.0, 1000),
            )

        # Verify commit
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 1

    def test_rollback_on_exception(self, temp_db):
        """Test that transaction rolls back on exception."""
        try:
            with temp_db.transaction() as conn:
                conn.execute(
                    "INSERT INTO documents(path, title, sha256, mtime, size) VALUES (?, ?, ?, ?, ?)",
                    ("/tmp/test.pdf", "Test", "abc123", 1234567890.0, 1000),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify rollback
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 0


class TestUpsertDocument:
    """Test document and chunks insertion/update."""

    def test_insert_new_document(self, temp_db):
        """Test inserting a new document."""
        doc = DocumentMetadata(
            path=Path("/tmp/test.pdf"),
            title="Test Document",
            sha256="abc123",
            mtime=1234567890.0,
            size=1000,
        )
        chunks = [
            ChunkRecord(document_path=doc.path, index=0, text="First chunk", metadata={"page": 1}),
            ChunkRecord(document_path=doc.path, index=1, text="Second chunk", metadata={"page": 2}),
        ]
        embeddings = np.random.rand(2, 384).astype("float32")

        status = temp_db.upsert_document(doc, chunks, embeddings)

        assert status == "inserted"

        # Verify document
        cursor = temp_db.connection.execute(
            "SELECT * FROM documents WHERE path = ?", (str(doc.path),)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["title"] == "Test Document"
        assert row["sha256"] == "abc123"

        # Verify chunks
        cursor = temp_db.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (row["id"],)
        )
        assert cursor.fetchone()[0] == 2

    def test_skip_unchanged_document(self, temp_db):
        """Test that unchanged documents are skipped."""
        doc = DocumentMetadata(
            path=Path("/tmp/test.pdf"), title="Test", sha256="abc123", mtime=1234567890.0, size=1000
        )
        chunks = [ChunkRecord(document_path=doc.path, index=0, text="Chunk", metadata={})]
        embeddings = np.random.rand(1, 384).astype("float32")

        # First insert
        status1 = temp_db.upsert_document(doc, chunks, embeddings)
        assert status1 == "inserted"

        # Second insert with same sha256
        status2 = temp_db.upsert_document(doc, chunks, embeddings)
        assert status2 == "skipped"

        # Verify only one document
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 1

    def test_update_modified_document(self, temp_db):
        """Test updating a document with different sha256."""
        doc1 = DocumentMetadata(
            path=Path("/tmp/test.pdf"),
            title="Test",
            sha256="old_hash",
            mtime=1234567890.0,
            size=1000,
        )
        chunks1 = [ChunkRecord(document_path=doc1.path, index=0, text="Old chunk", metadata={})]
        embeddings1 = np.random.rand(1, 384).astype("float32")

        # First insert
        status1 = temp_db.upsert_document(doc1, chunks1, embeddings1)
        assert status1 == "inserted"

        # Update with new sha256
        doc2 = DocumentMetadata(
            path=Path("/tmp/test.pdf"),
            title="Test Updated",
            sha256="new_hash",
            mtime=1234567891.0,
            size=1100,
        )
        chunks2 = [
            ChunkRecord(document_path=doc2.path, index=0, text="New chunk 1", metadata={}),
            ChunkRecord(document_path=doc2.path, index=1, text="New chunk 2", metadata={}),
        ]
        embeddings2 = np.random.rand(2, 384).astype("float32")

        status2 = temp_db.upsert_document(doc2, chunks2, embeddings2)
        assert status2 == "updated"

        # Verify document updated
        cursor = temp_db.connection.execute(
            "SELECT * FROM documents WHERE path = ?", (str(doc2.path),)
        )
        row = cursor.fetchone()
        assert row["sha256"] == "new_hash"
        assert row["title"] == "Test Updated"

        # Verify chunks replaced
        cursor = temp_db.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (row["id"],)
        )
        assert cursor.fetchone()[0] == 2

    def test_upsert_validation(self, temp_db):
        """Test validation of embeddings and chunks length."""
        doc = DocumentMetadata(
            path=Path("/tmp/test.pdf"), title="Test", sha256="abc123", mtime=1234567890.0, size=1000
        )
        chunks = [ChunkRecord(document_path=doc.path, index=0, text="Chunk", metadata={})]
        embeddings = np.random.rand(2, 384).astype("float32")  # Wrong size!

        with pytest.raises(ValueError, match="Embeddings and chunks length mismatch"):
            temp_db.upsert_document(doc, chunks, embeddings)

    def test_chunks_metadata_serialization(self, temp_db):
        """Test that chunk metadata is properly serialized to JSON."""
        doc = DocumentMetadata(
            path=Path("/tmp/test.pdf"), title="Test", sha256="abc123", mtime=1234567890.0, size=1000
        )
        chunks = [
            ChunkRecord(
                document_path=doc.path,
                index=0,
                text="Chunk",
                metadata={"page": 5, "section": "intro"},
            )
        ]
        embeddings = np.random.rand(1, 384).astype("float32")

        temp_db.upsert_document(doc, chunks, embeddings)

        cursor = temp_db.connection.execute("SELECT metadata FROM chunks")
        row = cursor.fetchone()
        metadata = json.loads(row["metadata"])
        assert metadata == {"page": 5, "section": "intro"}


class TestSearch:
    """Test vector search."""

    def test_search_empty_database(self, temp_db):
        """Test search on empty database."""
        query = np.random.rand(384).astype("float32")

        results = temp_db.search(query, top_k=5)

        assert results == []

    def test_search_returns_top_k(self, temp_db):
        """Test that search returns top_k results."""
        # Insert 10 documents
        for i in range(10):
            doc = DocumentMetadata(
                path=Path(f"/tmp/doc{i}.pdf"),
                title=f"Doc {i}",
                sha256=f"hash{i}",
                mtime=1234567890.0,
                size=1000,
            )
            chunks = [ChunkRecord(document_path=doc.path, index=0, text=f"Text {i}", metadata={})]
            embeddings = np.random.rand(1, 384).astype("float32")
            temp_db.upsert_document(doc, chunks, embeddings)

        query = np.random.rand(384).astype("float32")
        results = temp_db.search(query, top_k=5)

        assert len(results) == 5

    def test_search_returns_all_when_top_k_larger(self, temp_db):
        """Test that search returns all results when top_k > num_chunks."""
        # Insert 3 documents
        for i in range(3):
            doc = DocumentMetadata(
                path=Path(f"/tmp/doc{i}.pdf"),
                title=f"Doc {i}",
                sha256=f"hash{i}",
                mtime=1234567890.0,
                size=1000,
            )
            chunks = [ChunkRecord(document_path=doc.path, index=0, text=f"Text {i}", metadata={})]
            embeddings = np.random.rand(1, 384).astype("float32")
            temp_db.upsert_document(doc, chunks, embeddings)

        query = np.random.rand(384).astype("float32")
        results = temp_db.search(query, top_k=10)

        assert len(results) == 3

    def test_search_result_structure(self, temp_db):
        """Test structure of search results."""
        doc = DocumentMetadata(
            path=Path("/tmp/test.pdf"),
            title="Test Doc",
            sha256="abc123",
            mtime=1234567890.0,
            size=1000,
        )
        chunks = [
            ChunkRecord(document_path=doc.path, index=0, text="Sample text", metadata={"page": 1})
        ]
        embeddings = np.random.rand(1, 384).astype("float32")
        temp_db.upsert_document(doc, chunks, embeddings)

        query = np.random.rand(384).astype("float32")
        results = temp_db.search(query, top_k=1)

        assert len(results) == 1
        result = results[0]
        assert "path" in result
        assert "title" in result
        assert "chunk_index" in result
        assert "text" in result
        assert "metadata" in result
        assert "score" in result
        assert result["path"] == str(doc.path)
        assert result["title"] == "Test Doc"
        assert result["text"] == "Sample text"

    def test_search_scores_descending(self, temp_db):
        """Test that search results are sorted by score descending."""
        # Insert multiple documents
        for i in range(5):
            doc = DocumentMetadata(
                path=Path(f"/tmp/doc{i}.pdf"),
                title=f"Doc {i}",
                sha256=f"hash{i}",
                mtime=1234567890.0,
                size=1000,
            )
            chunks = [ChunkRecord(document_path=doc.path, index=0, text=f"Text {i}", metadata={})]
            embeddings = np.random.rand(1, 384).astype("float32")
            temp_db.upsert_document(doc, chunks, embeddings)

        query = np.random.rand(384).astype("float32")
        results = temp_db.search(query, top_k=5)

        # Verify scores are descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRemoveMissingFiles:
    """Test removing documents with missing files."""

    def test_remove_missing_files_empty_db(self, temp_db):
        """Test on empty database."""
        removed = temp_db.remove_missing_files()
        assert removed == 0

    def test_remove_missing_files(self, temp_db, tmp_path):
        """Test removing documents whose files don't exist."""
        # Create a real file
        real_file = tmp_path / "real.pdf"
        real_file.write_text("test")

        # Insert documents with real and fake paths
        doc_real = DocumentMetadata(
            path=real_file, title="Real", sha256="real_hash", mtime=1234567890.0, size=1000
        )
        doc_fake = DocumentMetadata(
            path=Path("/nonexistent/fake.pdf"),
            title="Fake",
            sha256="fake_hash",
            mtime=1234567890.0,
            size=1000,
        )

        chunks_real = [ChunkRecord(document_path=doc_real.path, index=0, text="Text", metadata={})]
        chunks_fake = [ChunkRecord(document_path=doc_fake.path, index=0, text="Text", metadata={})]
        embeddings = np.random.rand(1, 384).astype("float32")

        temp_db.upsert_document(doc_real, chunks_real, embeddings)
        temp_db.upsert_document(doc_fake, chunks_fake, embeddings)

        # Verify 2 documents
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 2

        # Remove missing
        removed = temp_db.remove_missing_files()

        assert removed == 1

        # Verify only real document remains
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM documents")
        assert cursor.fetchone()[0] == 1

        cursor = temp_db.connection.execute("SELECT path FROM documents")
        assert cursor.fetchone()["path"] == str(real_file)

    def test_remove_missing_cascades_chunks(self, temp_db):
        """Test that removing documents also removes chunks."""
        doc = DocumentMetadata(
            path=Path("/nonexistent/test.pdf"),
            title="Test",
            sha256="abc123",
            mtime=1234567890.0,
            size=1000,
        )
        chunks = [
            ChunkRecord(document_path=doc.path, index=0, text="Chunk 1", metadata={}),
            ChunkRecord(document_path=doc.path, index=1, text="Chunk 2", metadata={}),
        ]
        embeddings = np.random.rand(2, 384).astype("float32")

        temp_db.upsert_document(doc, chunks, embeddings)

        # Verify 2 chunks
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 2

        # Remove missing
        temp_db.remove_missing_files()

        # Verify chunks removed
        cursor = temp_db.connection.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 0
