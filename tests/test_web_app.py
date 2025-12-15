"""Tests for the FastAPI web application."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from docfinder.web.app import app, _resolve_db_path, _ensure_db_parent


client = TestClient(app)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_resolve_db_path_with_none(self) -> None:
        """Returns default path when db is None."""
        result = _resolve_db_path(None)
        assert isinstance(result, Path)

    def test_resolve_db_path_with_path(self, tmp_path: Path) -> None:
        """Returns resolved path when db is provided."""
        db_path = tmp_path / "custom.db"
        result = _resolve_db_path(db_path)
        assert result == db_path

    def test_ensure_db_parent_creates_directory(self, tmp_path: Path) -> None:
        """Creates parent directory if it doesn't exist."""
        db_path = tmp_path / "subdir" / "test.db"
        assert not db_path.parent.exists()
        _ensure_db_parent(db_path)
        assert db_path.parent.exists()


class TestSearchEndpoint:
    """Tests for POST /search endpoint."""

    def test_search_empty_query(self) -> None:
        """Returns 400 for empty query."""
        response = client.post("/search", json={"query": "", "top_k": 10})
        assert response.status_code == 400
        assert "Empty query" in response.json()["detail"]

    def test_search_whitespace_query(self) -> None:
        """Returns 400 for whitespace-only query."""
        response = client.post("/search", json={"query": "   ", "top_k": 10})
        assert response.status_code == 400
        assert "Empty query" in response.json()["detail"]

    def test_search_database_not_found(self, tmp_path: Path) -> None:
        """Returns 404 when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        response = client.post(
            "/search", json={"query": "test", "db": str(db_path), "top_k": 10}
        )
        assert response.status_code == 404
        assert "Database not found" in response.json()["detail"]

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    @patch("docfinder.web.app.Searcher")
    def test_search_success(
        self,
        mock_searcher_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Returns search results on success."""
        from docfinder.index.search import SearchResult

        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        # Use real SearchResult instead of MagicMock
        real_result = SearchResult(
            score=0.95,
            path="/path/to/doc.pdf",
            chunk_index=0,
            text="Test content",
            title="Test Document",
            metadata={},
        )

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [real_result]
        mock_searcher_class.return_value = mock_searcher

        response = client.post(
            "/search", json={"query": "test", "db": str(db_path), "top_k": 10}
        )
        assert response.status_code == 200
        assert "results" in response.json()

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    @patch("docfinder.web.app.Searcher")
    def test_search_clamps_top_k(
        self,
        mock_searcher_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Clamps top_k to valid range."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        # Test with top_k > 50
        response = client.post(
            "/search", json={"query": "test", "db": str(db_path), "top_k": 100}
        )
        assert response.status_code == 200
        mock_searcher.search.assert_called_with("test", top_k=50)


class TestOpenEndpoint:
    """Tests for POST /open endpoint."""

    def test_open_file_not_found(self, tmp_path: Path) -> None:
        """Returns 404 when file doesn't exist."""
        response = client.post("/open", json={"path": str(tmp_path / "nonexistent.pdf")})
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    @patch("subprocess.Popen")
    def test_open_file_success_posix(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        """Opens file on POSIX systems."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        with patch("os.name", "posix"):
            with patch("sys.platform", "darwin"):
                response = client.post("/open", json={"path": str(test_file)})
                assert response.status_code == 200
                assert response.json()["status"] == "ok"


class TestDocumentsEndpoint:
    """Tests for GET /documents endpoint."""

    def test_documents_database_not_found(self, tmp_path: Path) -> None:
        """Returns empty list when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        response = client.get(f"/documents?db={db_path}")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["stats"]["document_count"] == 0

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_documents_success(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Returns document list on success."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.list_documents.return_value = [
            {"id": 1, "path": "/doc.pdf", "title": "Test"}
        ]
        mock_store.get_stats.return_value = {
            "document_count": 1,
            "chunk_count": 5,
            "total_size_bytes": 1024,
        }
        mock_store_class.return_value = mock_store

        response = client.get(f"/documents?db={db_path}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["stats"]["document_count"] == 1


class TestDeleteDocumentEndpoint:
    """Tests for DELETE /documents/{doc_id} endpoint."""

    def test_delete_database_not_found(self, tmp_path: Path) -> None:
        """Returns 404 when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        response = client.delete(f"/documents/1?db={db_path}")
        assert response.status_code == 404
        assert "Database not found" in response.json()["detail"]

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_delete_document_not_found(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Returns 404 when document doesn't exist."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.delete_document.return_value = False
        mock_store_class.return_value = mock_store

        response = client.delete(f"/documents/999?db={db_path}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_delete_document_success(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successfully deletes document."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.delete_document.return_value = True
        mock_store_class.return_value = mock_store

        response = client.delete(f"/documents/1?db={db_path}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDeleteDocumentByPathEndpoint:
    """Tests for POST /documents/delete endpoint."""

    def test_delete_no_identifier(self) -> None:
        """Returns 400 when neither doc_id nor path provided."""
        response = client.post("/documents/delete", json={})
        assert response.status_code == 400
        assert "Either doc_id or path" in response.json()["detail"]

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_delete_by_path_success(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successfully deletes document by path."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.delete_document_by_path.return_value = True
        mock_store_class.return_value = mock_store

        response = client.post(
            f"/documents/delete?db={db_path}", json={"path": "/path/to/doc.pdf"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCleanupEndpoint:
    """Tests for DELETE /documents/cleanup endpoint."""

    def test_cleanup_database_not_found(self, tmp_path: Path) -> None:
        """Returns 404 when database doesn't exist."""
        from urllib.parse import quote

        db_path = tmp_path / "nonexistent.db"
        # URL encode the path to handle special characters
        encoded_path = quote(str(db_path), safe="")
        response = client.delete(f"/documents/cleanup?db={encoded_path}")
        assert response.status_code == 404
        assert "Database not found" in response.json()["detail"]

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_cleanup_success(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successfully removes missing files."""
        from urllib.parse import quote

        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.remove_missing_files.return_value = 2
        mock_store_class.return_value = mock_store

        encoded_path = quote(str(db_path), safe="")
        response = client.delete(f"/documents/cleanup?db={encoded_path}")
        assert response.status_code == 200
        assert response.json()["removed_count"] == 2


class TestIndexEndpoint:
    """Tests for POST /index endpoint."""

    def test_index_no_paths(self) -> None:
        """Returns 400 when no paths provided."""
        response = client.post("/index", json={"paths": []})
        assert response.status_code == 400
        assert "No path provided" in response.json()["detail"]

    def test_index_null_byte_in_path(self) -> None:
        """Returns 400 for path with null byte."""
        response = client.post("/index", json={"paths": ["/path/with\x00null"]})
        assert response.status_code == 400
        assert "null byte" in response.json()["detail"]

    def test_index_path_not_found(self, tmp_path: Path) -> None:
        """Returns error for nonexistent path (403 if outside home, 404 if inside)."""
        # Create path inside home directory that doesn't exist
        fake_path = Path.home() / "docfinder_test_nonexistent_12345"
        response = client.post("/index", json={"paths": [str(fake_path)]})
        # Should be 404 because path is inside home but doesn't exist
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_index_path_not_directory(self, tmp_path: Path) -> None:
        """Returns 400 when path is a file, not directory."""
        # Create file inside home directory
        file_path = Path.home() / "docfinder_test_file_12345.txt"
        file_path.touch()
        try:
            response = client.post("/index", json={"paths": [str(file_path)]})
            assert response.status_code == 400
            assert "must be a directory" in response.json()["detail"]
        finally:
            file_path.unlink()

    def test_index_path_outside_home(self, tmp_path: Path) -> None:
        """Returns 403 for path outside home directory."""
        # Try to access /etc which is definitely outside home
        response = client.post("/index", json={"paths": ["/etc"]})
        assert response.status_code == 403
        assert "outside allowed directory" in response.json()["detail"]

    @patch("docfinder.web.app._run_index_job")
    def test_index_success(
        self, mock_run_index: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successfully indexes directory."""
        # Create a subdirectory inside home
        test_dir = Path.home() / "DocFinder_test_temp"
        test_dir.mkdir(exist_ok=True)

        try:
            mock_run_index.return_value = {
                "inserted": 1,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "processed_files": [],
            }

            response = client.post("/index", json={"paths": [str(test_dir)]})
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
        finally:
            test_dir.rmdir()


class TestFrontendRouter:
    """Tests for the frontend HTML routes."""

    def test_index_page(self) -> None:
        """Returns HTML for index page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "DocFinder" in response.text
