"""Tests for core data models."""

from __future__ import annotations

from pathlib import Path

import pytest

from docfinder.models import ChunkRecord, DocumentMetadata


class TestDocumentMetadata:
    """Test DocumentMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Should create DocumentMetadata with all fields."""
        metadata = DocumentMetadata(
            path=Path("/path/to/doc.pdf"),
            title="Test Document",
            sha256="abc123",
            mtime=1234567890.0,
            size=1024,
        )
        
        assert metadata.path == Path("/path/to/doc.pdf")
        assert metadata.title == "Test Document"
        assert metadata.sha256 == "abc123"
        assert metadata.mtime == 1234567890.0
        assert metadata.size == 1024

    def test_metadata_equality(self) -> None:
        """Should compare metadata by value."""
        meta1 = DocumentMetadata(
            path=Path("/test.pdf"),
            title="Test",
            sha256="hash1",
            mtime=123.0,
            size=100,
        )
        meta2 = DocumentMetadata(
            path=Path("/test.pdf"),
            title="Test",
            sha256="hash1",
            mtime=123.0,
            size=100,
        )
        
        assert meta1 == meta2

    def test_metadata_inequality(self) -> None:
        """Should detect differences in metadata."""
        meta1 = DocumentMetadata(
            path=Path("/test1.pdf"),
            title="Test1",
            sha256="hash1",
            mtime=123.0,
            size=100,
        )
        meta2 = DocumentMetadata(
            path=Path("/test2.pdf"),
            title="Test2",
            sha256="hash2",
            mtime=456.0,
            size=200,
        )
        
        assert meta1 != meta2


class TestChunkRecord:
    """Test ChunkRecord dataclass."""

    def test_create_chunk(self) -> None:
        """Should create ChunkRecord with all fields."""
        chunk = ChunkRecord(
            document_path=Path("/path/to/doc.pdf"),
            index=0,
            text="This is a test chunk",
            metadata={"page": 1, "section": "intro"},
        )
        
        assert chunk.document_path == Path("/path/to/doc.pdf")
        assert chunk.index == 0
        assert chunk.text == "This is a test chunk"
        assert chunk.metadata == {"page": 1, "section": "intro"}

    def test_chunk_with_empty_metadata(self) -> None:
        """Should handle empty metadata dict."""
        chunk = ChunkRecord(
            document_path=Path("/doc.pdf"),
            index=5,
            text="Chunk text",
            metadata={},
        )
        
        assert chunk.metadata == {}

    def test_chunk_equality(self) -> None:
        """Should compare chunks by value."""
        chunk1 = ChunkRecord(
            document_path=Path("/test.pdf"),
            index=0,
            text="Text",
            metadata={"k": "v"},
        )
        chunk2 = ChunkRecord(
            document_path=Path("/test.pdf"),
            index=0,
            text="Text",
            metadata={"k": "v"},
        )
        
        assert chunk1 == chunk2

    def test_chunk_different_index(self) -> None:
        """Should differentiate chunks by index."""
        chunk1 = ChunkRecord(
            document_path=Path("/test.pdf"),
            index=0,
            text="Same text",
            metadata={},
        )
        chunk2 = ChunkRecord(
            document_path=Path("/test.pdf"),
            index=1,
            text="Same text",
            metadata={},
        )
        
        assert chunk1 != chunk2
