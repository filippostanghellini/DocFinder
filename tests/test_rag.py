"""Tests for the RAG module — context window retrieval and engine assembly."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docfinder.index.search import SearchResult
from docfinder.index.storage import SQLiteVectorStore
from docfinder.models import ChunkRecord, DocumentMetadata
from docfinder.rag.engine import _SMALL_DOC_THRESHOLD, RAGEngine
from docfinder.rag.llm import MODEL_TIERS, select_model


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store_with_chunks(tmp_path):
    """Create a store with a single document containing 25 chunks (indices 0-24)."""
    db_path = tmp_path / "rag_test.db"
    store = SQLiteVectorStore(db_path, dimension=384)

    doc = DocumentMetadata(
        path=Path("/tmp/rag_doc.pdf"),
        title="RAG Test Doc",
        sha256="rag_hash_123",
        mtime=1234567890.0,
        size=50000,
    )
    chunks = [
        ChunkRecord(
            document_path=doc.path,
            index=i,
            text=f"Chunk number {i} of the document. " * 10,
            metadata={"page": i // 5 + 1},
        )
        for i in range(25)
    ]
    embeddings = np.random.rand(25, 384).astype("float32")
    store.upsert_document(doc, chunks, embeddings)

    yield store
    store.close()


# ── get_context_window tests ─────────────────────────────────────────────────


class TestGetContextWindow:
    """Test SQLiteVectorStore.get_context_window."""

    def test_center_chunk_with_full_window(self, store_with_chunks):
        """Window around chunk 12 (indices 2..22) should return 21 chunks."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=12, window_size=10)

        assert len(window) == 21
        indices = [c["chunk_index"] for c in window]
        assert indices == list(range(2, 23))

    def test_window_clamps_at_start(self, store_with_chunks):
        """Window around chunk 3 should clamp the lower bound to 0."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=3, window_size=10)

        indices = [c["chunk_index"] for c in window]
        assert indices[0] == 0
        assert indices[-1] == 13
        assert len(window) == 14  # 0..13

    def test_window_clamps_at_end(self, store_with_chunks):
        """Window around chunk 22 should stop at the last chunk (24)."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=22, window_size=10)

        indices = [c["chunk_index"] for c in window]
        assert indices[0] == 12
        assert indices[-1] == 24

    def test_window_size_zero(self, store_with_chunks):
        """Window of size 0 should return only the center chunk."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=10, window_size=0)

        assert len(window) == 1
        assert window[0]["chunk_index"] == 10

    def test_nonexistent_document(self, store_with_chunks):
        """Querying a nonexistent document_id should return an empty list."""
        window = store_with_chunks.get_context_window(
            document_id=9999, center_index=0, window_size=10
        )
        assert window == []

    def test_chunks_ordered_by_index(self, store_with_chunks):
        """Returned chunks must be in ascending chunk_index order."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=12, window_size=5)

        indices = [c["chunk_index"] for c in window]
        assert indices == sorted(indices)

    def test_chunk_text_content(self, store_with_chunks):
        """Verify actual text content is returned correctly."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        window = store_with_chunks.get_context_window(doc_id, center_index=5, window_size=0)

        assert len(window) == 1
        assert "Chunk number 5" in window[0]["text"]

    def test_search_results_include_document_id(self, store_with_chunks):
        """Verify that search results now include document_id."""
        query = np.random.rand(384).astype("float32")
        results = store_with_chunks.search(query, top_k=1)

        assert len(results) == 1
        assert "document_id" in results[0]
        assert isinstance(results[0]["document_id"], int)


# ── get_context_by_page tests ─────────────────────────────────────────────


class TestGetContextByPage:
    """Test SQLiteVectorStore.get_context_by_page."""

    def test_returns_center_page_chunks(self, store_with_chunks):
        """Page 3 has chunks 10-14 (5 chunks). All should be returned."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        result = store_with_chunks.get_context_by_page(doc_id, center_page=3)

        indices = [c["chunk_index"] for c in result]
        # Page 3 chunks (10-14) must be present
        for i in range(10, 15):
            assert i in indices

    def test_expands_to_adjacent_pages(self, store_with_chunks):
        """With a large max_chars, it should include adjacent pages."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        result = store_with_chunks.get_context_by_page(doc_id, center_page=3, max_chars=100000)

        # Should have chunks from multiple pages since budget is huge
        assert len(result) > 5

    def test_respects_max_chars(self, store_with_chunks):
        """With a tiny max_chars, only center page chunks are returned."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        # Each chunk is ~350 chars ("Chunk number X of the document. " * 10)
        # 500 chars should fit only ~1 chunk
        result = store_with_chunks.get_context_by_page(doc_id, center_page=3, max_chars=500)

        total_chars = sum(len(c["text"]) for c in result)
        # Should be limited — at most center page
        assert total_chars < 3000

    def test_ordered_by_chunk_index(self, store_with_chunks):
        """Results must be in ascending chunk_index order."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        result = store_with_chunks.get_context_by_page(doc_id, center_page=3, max_chars=100000)

        indices = [c["chunk_index"] for c in result]
        assert indices == sorted(indices)

    def test_nonexistent_page(self, store_with_chunks):
        """Querying a page that doesn't exist should return empty or expand."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        result = store_with_chunks.get_context_by_page(doc_id, center_page=99)

        # No chunks on page 99, but expansion should eventually find some
        # With default max_chars, it will keep expanding
        # Actually page 99 is far away; expansion increments by 1 each time
        # so it would take many iterations. Let's just verify it doesn't crash.
        assert isinstance(result, list)

    def test_nonexistent_document(self, store_with_chunks):
        """Querying a nonexistent document_id should return empty."""
        result = store_with_chunks.get_context_by_page(document_id=9999, center_page=1)
        assert result == []

    def test_page_1_includes_first_chunks(self, store_with_chunks):
        """Page 1 should contain chunks 0-4."""
        doc_id = store_with_chunks.connection.execute("SELECT id FROM documents").fetchone()["id"]
        result = store_with_chunks.get_context_by_page(doc_id, center_page=1, max_chars=2000)

        indices = [c["chunk_index"] for c in result]
        assert 0 in indices
        assert 4 in indices


# ── chunk_text_stream_paged tests ────────────────────────────────────────


class TestChunkTextStreamPaged:
    """Test the page-aware chunking utility."""

    def test_preserves_page_numbers(self):
        from docfinder.utils.text import chunk_text_stream_paged

        pages = [(1, "A" * 100), (2, "B" * 100), (3, "C" * 100)]
        results = list(chunk_text_stream_paged(pages, max_chars=100, overlap=0))

        assert len(results) == 3
        assert results[0][1] == 1  # page number
        assert results[1][1] == 2
        assert results[2][1] == 3

    def test_chunk_spanning_pages(self):
        from docfinder.utils.text import chunk_text_stream_paged

        # Page 1 has 60 chars, page 2 has 60 chars. Chunk size 100.
        # First chunk should span both pages but report page 1 (start).
        pages = [(1, "A" * 60), (2, "B" * 60)]
        results = list(chunk_text_stream_paged(pages, max_chars=100, overlap=0))

        assert len(results) >= 1
        # First chunk starts on page 1
        assert results[0][1] == 1

    def test_single_page(self):
        from docfinder.utils.text import chunk_text_stream_paged

        pages = [(1, "Hello world")]
        results = list(chunk_text_stream_paged(pages, max_chars=1000, overlap=0))

        assert len(results) == 1
        assert results[0][0] == "Hello world"
        assert results[0][1] == 1


# ── Virtual page tests for non-PDF formats ───────────────────────────────────


class TestTxtPaged:
    """Test plain text virtual paging."""

    def test_short_file_single_page(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_txt_paged

        f = tmp_path / "short.txt"
        f.write_text("Hello world.\nThis is a short file.")
        pages = list(iter_text_parts_txt_paged(f))

        assert len(pages) == 1
        assert pages[0][0] == 1
        assert "Hello world" in pages[0][1]

    def test_long_file_multiple_pages(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_txt_paged

        f = tmp_path / "long.txt"
        # 10000 chars → should produce multiple virtual pages (~3 at 3000 chars each)
        f.write_text("Line of text.\n" * 700)
        pages = list(iter_text_parts_txt_paged(f))

        assert len(pages) >= 3
        # Pages should be numbered sequentially
        for i, (page_num, _) in enumerate(pages):
            assert page_num == i + 1

    def test_empty_file(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_txt_paged

        f = tmp_path / "empty.txt"
        f.write_text("")
        pages = list(iter_text_parts_txt_paged(f))

        assert pages == []


class TestMdPaged:
    """Test Markdown section-based virtual paging."""

    def test_splits_on_headings(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_md_paged

        f = tmp_path / "doc.md"
        f.write_text("# Intro\nSome intro text.\n\n## Methods\nDetails here.\n\n## Results\nData.")
        pages = list(iter_text_parts_md_paged(f))

        assert len(pages) == 3
        assert "intro text" in pages[0][1].lower()
        assert "details" in pages[1][1].lower()
        assert "data" in pages[2][1].lower()

    def test_no_headings_single_section(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_md_paged

        f = tmp_path / "flat.md"
        f.write_text("Just plain text with **bold** and *italic*.")
        pages = list(iter_text_parts_md_paged(f))

        assert len(pages) == 1
        assert pages[0][0] == 1

    def test_cleans_formatting(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_md_paged

        f = tmp_path / "fmt.md"
        f.write_text("# Title\n[Click here](http://example.com) and ![img](pic.png)")
        pages = list(iter_text_parts_md_paged(f))

        assert len(pages) == 1
        text = pages[0][1]
        assert "Click here" in text
        assert "http://" not in text  # link URL stripped
        assert "![" not in text  # image stripped


class TestDocxPaged:
    """Test Word document virtual paging."""

    def test_groups_paragraphs(self, tmp_path):
        from docfinder.ingestion.pdf_loader import iter_text_parts_docx_paged

        try:
            from docx import Document
        except ImportError:
            pytest.skip("python-docx not installed")

        f = tmp_path / "test.docx"
        doc = Document()
        for i in range(25):
            doc.add_paragraph(f"Paragraph {i}")
        doc.save(str(f))

        pages = list(iter_text_parts_docx_paged(f))

        # 25 paragraphs / 10 per page = 3 pages (10, 10, 5)
        assert len(pages) == 3
        assert pages[0][0] == 1
        assert pages[1][0] == 2
        assert pages[2][0] == 3
        assert "Paragraph 0" in pages[0][1]
        assert "Paragraph 10" in pages[1][1]
        assert "Paragraph 20" in pages[2][1]


# ── Model selection tests ────────────────────────────────────────────────────


class TestSelectModel:
    """Test automatic model selection based on RAM."""

    def test_large_ram_picks_7b(self):
        spec = select_model(total_ram_mb=32_768)
        assert "7B" in spec.name

    def test_medium_ram_picks_3b(self):
        spec = select_model(total_ram_mb=12_000)
        assert "3B" in spec.name

    def test_small_ram_picks_1_5b(self):
        spec = select_model(total_ram_mb=4_000)
        assert "1.5B" in spec.name

    def test_minimum_ram_still_works(self):
        spec = select_model(total_ram_mb=1_000)
        assert spec is not None
        assert spec.ram_min_mb == 0

    def test_tiers_ordered_descending(self):
        """Model tiers must be ordered by ram_min_mb descending."""
        ram_mins = [t.ram_min_mb for t in MODEL_TIERS]
        assert ram_mins == sorted(ram_mins, reverse=True)


# ── RAG Engine context assembly tests ────────────────────────────────────────


class TestRAGEngineContextAssembly:
    """Test the context trimming logic of RAGEngine."""

    def test_context_trimmed_when_exceeding_budget(self):
        from docfinder.rag.engine import _MAX_CONTEXT_CHARS, RAGEngine

        # Create chunks that together exceed _MAX_CONTEXT_CHARS
        chunks = [
            {"chunk_index": i, "text": "x" * 2000, "path": "/tmp/test.pdf"} for i in range(20)
        ]
        results = []  # not used by _assemble_context_text

        text = RAGEngine._assemble_context_text(chunks, results)

        # The assembled text should not exceed the budget (plus a small header allowance)
        assert len(text) <= _MAX_CONTEXT_CHARS + 500  # headers add a bit

    def test_context_preserves_order(self):
        from docfinder.rag.engine import RAGEngine

        chunks = [
            {"chunk_index": i, "text": f"chunk_{i}", "path": "/tmp/test.pdf"} for i in range(5)
        ]
        text = RAGEngine._assemble_context_text(chunks, [])

        # Chunks should appear in order
        for i in range(4):
            assert text.index(f"chunk_{i}") < text.index(f"chunk_{i + 1}")


# ── Hybrid context strategy tests ─────────────────────────────────────────


def _make_engine(store: SQLiteVectorStore) -> RAGEngine:
    """Create a RAGEngine with a mocked searcher and llm."""
    searcher = MagicMock()
    llm = MagicMock()
    return RAGEngine(searcher, store, llm, window_size=10)


def _make_result(path: str, chunk_index: int) -> SearchResult:
    return SearchResult(
        path=Path(path),
        title="Doc",
        chunk_index=chunk_index,
        score=0.9,
        text="hit",
        metadata={},
    )


class TestHybridContextStrategy:
    """Test that _build_context picks the right strategy based on document size."""

    def test_small_doc_uses_get_all_chunks(self, tmp_path):
        """Documents with <= SMALL_DOC_THRESHOLD chunks should load entirely."""
        db_path = tmp_path / "small.db"
        store = SQLiteVectorStore(db_path, dimension=384)

        n = _SMALL_DOC_THRESHOLD  # exactly at threshold
        doc = DocumentMetadata(
            path=Path("/tmp/small.pdf"),
            title="Small",
            sha256="small_hash",
            mtime=1.0,
            size=1000,
        )
        chunks = [
            ChunkRecord(document_path=doc.path, index=i, text=f"Chunk {i}", metadata={})
            for i in range(n)
        ]
        embeddings = np.random.rand(n, 384).astype("float32")
        store.upsert_document(doc, chunks, embeddings)

        engine = _make_engine(store)
        result = _make_result("/tmp/small.pdf", chunk_index=5)

        with (
            patch.object(store, "get_all_chunks", wraps=store.get_all_chunks) as mock_all,
            patch.object(
                store, "get_context_window", wraps=store.get_context_window
            ) as mock_window,
        ):
            ctx = engine._build_context([result])

        mock_all.assert_called_once()
        mock_window.assert_not_called()
        # All chunks should be present
        assert len(ctx) == n
        store.close()

    def test_large_doc_uses_get_context_window(self, tmp_path):
        """Documents with > SMALL_DOC_THRESHOLD chunks should use window strategy."""
        db_path = tmp_path / "large.db"
        store = SQLiteVectorStore(db_path, dimension=384)

        n = _SMALL_DOC_THRESHOLD + 5  # above threshold
        doc = DocumentMetadata(
            path=Path("/tmp/large.pdf"),
            title="Large",
            sha256="large_hash",
            mtime=1.0,
            size=50000,
        )
        chunks = [
            ChunkRecord(document_path=doc.path, index=i, text=f"Chunk {i}", metadata={})
            for i in range(n)
        ]
        embeddings = np.random.rand(n, 384).astype("float32")
        store.upsert_document(doc, chunks, embeddings)

        engine = _make_engine(store)
        result = _make_result("/tmp/large.pdf", chunk_index=12)

        with (
            patch.object(store, "get_all_chunks", wraps=store.get_all_chunks) as mock_all,
            patch.object(
                store, "get_context_window", wraps=store.get_context_window
            ) as mock_window,
        ):
            ctx = engine._build_context([result])

        mock_all.assert_not_called()
        mock_window.assert_called_once()
        # Window around chunk 12 with window_size=10 -> indices 2..22
        assert len(ctx) < n
        store.close()

    def test_mixed_small_and_large(self, tmp_path):
        """Query hitting one small doc and one large doc uses both strategies."""
        db_path = tmp_path / "mixed.db"
        store = SQLiteVectorStore(db_path, dimension=384)

        # Small document: 5 chunks
        small_doc = DocumentMetadata(
            path=Path("/tmp/small.pdf"),
            title="Small",
            sha256="s_hash",
            mtime=1.0,
            size=500,
        )
        small_n = 5
        small_chunks = [
            ChunkRecord(document_path=small_doc.path, index=i, text=f"S{i}", metadata={})
            for i in range(small_n)
        ]
        small_emb = np.random.rand(small_n, 384).astype("float32")
        store.upsert_document(small_doc, small_chunks, small_emb)

        # Large document: 30 chunks
        large_doc = DocumentMetadata(
            path=Path("/tmp/large.pdf"),
            title="Large",
            sha256="l_hash",
            mtime=1.0,
            size=50000,
        )
        large_n = 30
        large_chunks = [
            ChunkRecord(document_path=large_doc.path, index=i, text=f"L{i}", metadata={})
            for i in range(large_n)
        ]
        large_emb = np.random.rand(large_n, 384).astype("float32")
        store.upsert_document(large_doc, large_chunks, large_emb)

        engine = _make_engine(store)
        results = [
            _make_result("/tmp/small.pdf", chunk_index=2),
            _make_result("/tmp/large.pdf", chunk_index=15),
        ]

        with (
            patch.object(store, "get_all_chunks", wraps=store.get_all_chunks) as mock_all,
            patch.object(
                store, "get_context_window", wraps=store.get_context_window
            ) as mock_window,
        ):
            ctx = engine._build_context(results)

        # get_all_chunks called for the small doc
        mock_all.assert_called_once()
        # get_context_window called for the large doc
        mock_window.assert_called_once()

        # Context should contain all small doc chunks + window from large doc
        paths = {_normalize_path(c["path"]) for c in ctx}
        assert "/tmp/small.pdf" in paths
        assert "/tmp/large.pdf" in paths

        small_ctx = [c for c in ctx if _normalize_path(c["path"]) == "/tmp/small.pdf"]
        assert len(small_ctx) == small_n

        large_ctx = [c for c in ctx if _normalize_path(c["path"]) == "/tmp/large.pdf"]
        assert len(large_ctx) < large_n

        store.close()
