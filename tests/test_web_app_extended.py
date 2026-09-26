"""Extended tests for web app endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from docfinder.index.storage import SQLiteVectorStore
from docfinder.ollama import OllamaError
from docfinder.web.app import app

client = TestClient(app)


class TestSettingsEndpoints:
    """Tests for GET/POST /settings endpoints."""

    @patch("docfinder.web.app.load_settings")
    def test_get_settings(self, mock_load_settings: MagicMock) -> None:
        """Returns current settings."""
        mock_load_settings.return_value = {"hotkey": "<cmd>+space", "hotkey_enabled": True}
        response = client.get("/settings")
        assert response.status_code == 200
        assert response.json()["hotkey"] == "<cmd>+space"
        assert response.json()["hotkey_enabled"] is True

    @patch("docfinder.web.app.load_settings")
    @patch("docfinder.web.app._save_settings")
    def test_update_settings(self, mock_save: MagicMock, mock_load: MagicMock) -> None:
        """Updates and returns settings."""
        mock_load.return_value = {"hotkey": "<alt>+d", "hotkey_enabled": True}
        response = client.post("/settings", json={"hotkey": "<cmd>+f"})
        assert response.status_code == 200
        assert response.json()["hotkey"] == "<cmd>+f"


class TestRAGEndpoints:
    """Tests for RAG endpoints."""

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_rag_models_no_db(
        self, mock_store_class: MagicMock, mock_embedder_class: MagicMock
    ) -> None:
        """Returns model list even without database."""
        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        response = client.get("/rag/models")
        assert response.status_code == 200
        assert "models" in response.json()
        assert "total_ram_mb" in response.json()

    def test_rag_download_already_running(self) -> None:
        """Returns already_running when download in progress."""
        import docfinder.web.app as web_app

        original_status = web_app._rag_download["status"]
        web_app._rag_download["status"] = "downloading"
        try:
            response = client.post("/rag/download", params={"model_name": "test"})
            assert response.status_code == 200
            assert response.json()["status"] == "already_running"
        finally:
            web_app._rag_download["status"] = original_status

    def test_rag_chat_no_model_loaded(self) -> None:
        """Returns 503 when RAG model not loaded."""
        from docfinder.web.app import _rag_llm

        # Ensure no model is loaded
        original = _rag_llm
        import docfinder.web.app as web_app

        web_app._rag_llm = None
        try:
            response = client.post(
                "/rag/chat",
                json={
                    "question": "test?",
                    "document_path": "/doc.pdf",
                    "chunk_index": 0,
                },
            )
            assert response.status_code == 503
        finally:
            web_app._rag_llm = original


class TestIndexStatusEndpoint:
    """Tests for /index/status/{job_id} endpoint."""

    def test_job_not_found(self) -> None:
        """Returns 404 for unknown job ID."""
        response = client.get("/index/status/nonexistent-job-id")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]


class TestScanEndpoint:
    """Tests for POST /index/scan endpoint."""

    def test_scan_no_paths(self) -> None:
        """Returns 400 when no paths provided."""
        response = client.post("/index/scan", json={"paths": []})
        assert response.status_code == 400
        assert "No path provided" in response.json()["detail"]


class TestSystemInfo:
    """Tests for /system/info endpoint."""

    @patch("docfinder.web.app._get_memory_info")
    @patch("docfinder.web.app._get_runtime_info")
    def test_system_info_combines_memory_and_runtime(
        self, mock_runtime: MagicMock, mock_memory: MagicMock
    ) -> None:
        """Combines memory and runtime info."""
        mock_memory.return_value = {"available_mb": 8192, "total_mb": 16384}
        mock_runtime.return_value = {"selected_backend": "onnx", "selected_device": "cpu"}
        response = client.get("/system/info")
        assert response.status_code == 200
        assert response.json()["available_mb"] == 8192
        assert response.json()["selected_backend"] == "onnx"


class TestSpotlightHide:
    """Tests for POST /gui/spotlight/hide endpoint."""

    def test_spotlight_hide_no_callback(self) -> None:
        """Returns ok even without callback registered."""
        from docfinder.web.app import _spotlight_hide_callback

        original = _spotlight_hide_callback
        import docfinder.web.app as web_app

        web_app._spotlight_hide_callback = None
        try:
            response = client.post("/gui/spotlight/hide")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
        finally:
            web_app._spotlight_hide_callback = original


class TestOllamaModelsEndpoint:
    """Tests for GET /api/ollama/models endpoint."""

    @patch("docfinder.web.app.list_ollama_models")
    def test_connected(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [{"name": "llama3", "size_bytes": 123}]
        response = client.get("/api/ollama/models?url=http://x")
        assert response.status_code == 200
        assert response.json() == {
            "connected": True,
            "models": [{"name": "llama3", "size_bytes": 123}],
            "error": None,
        }

    @patch("docfinder.web.app.list_ollama_models")
    def test_unreachable_never_raises(self, mock_list: MagicMock) -> None:
        mock_list.side_effect = OllamaError("Ollama server unreachable at http://x: boom")
        response = client.get("/api/ollama/models?url=http://x")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert "unreachable" in data["error"]

    def test_no_url_configured(self) -> None:
        response = client.get("/api/ollama/models")
        assert response.status_code == 200
        assert response.json()["connected"] is False


class TestDeleteAllDocumentsEndpoint:
    """Tests for DELETE /documents/all endpoint."""

    @patch("docfinder.web.app.EmbeddingModel")
    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_clears_store(
        self, mock_store_class: MagicMock, mock_embedder_class: MagicMock, tmp_path
    ) -> None:
        db_path = tmp_path / "test.db"
        db_path.touch()
        mock_embedder_class.return_value = MagicMock(dimension=768)
        mock_store = MagicMock()
        mock_store.clear_all.return_value = 5
        mock_store_class.return_value = mock_store

        response = client.delete(f"/documents/delete-all?db={db_path}")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "removed": 5}

    def test_database_not_found(self, tmp_path) -> None:
        response = client.delete(f"/documents/delete-all?db={tmp_path / 'missing.db'}")
        assert response.status_code == 404


class TestReindexEndpoint:
    """Tests for POST /index/reindex endpoint."""

    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_no_source_paths(self, mock_store_class: MagicMock, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        db_path.touch()
        mock_store = MagicMock()
        mock_store.get_meta.return_value = None
        mock_store_class.return_value = mock_store

        response = client.post(f"/index/reindex?db={db_path}")
        assert response.status_code == 400
        assert "No source paths" in response.json()["detail"]

    @patch("docfinder.web.app.SQLiteVectorStore")
    def test_starts_index_job(self, mock_store_class: MagicMock, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        db_path.touch()
        mock_store = MagicMock()
        mock_store.get_meta.return_value = json.dumps([str(tmp_path)])
        mock_store_class.return_value = mock_store

        with patch("docfinder.web.app._validate_paths") as mock_validate:
            mock_validate.return_value = [tmp_path]
            with patch("docfinder.web.app._run_index_job") as mock_run:
                mock_run.return_value = {"inserted": 1, "updated": 0, "skipped": 0, "failed": 0}
                response = client.post(f"/index/reindex?db={db_path}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "job_id" in response.json()


class TestSettingsOllamaFields:
    """Tests for Ollama settings persistence via POST /settings."""

    @patch("docfinder.web.app._save_settings")
    @patch("docfinder.web.app.load_settings")
    def test_saves_ollama_fields(self, mock_load: MagicMock, mock_save: MagicMock) -> None:
        mock_load.return_value = {"hotkey": "<alt>+d", "hotkey_enabled": True}
        response = client.post(
            "/settings",
            json={
                "embedding_backend": "ollama",
                "embedding_model": "nomic-embed-text",
                "ollama_url": "http://127.0.0.1:11434",
                "llm_backend": "ollama",
                "llm_model": "llama3",
            },
        )
        assert response.status_code == 200
        saved = mock_save.call_args[0][0]
        assert saved["embedding_backend"] == "ollama"
        assert saved["embedding_model"] == "nomic-embed-text"
        assert saved["ollama_url"] == "http://127.0.0.1:11434"
        assert saved["llm_backend"] == "ollama"
        assert saved["llm_model"] == "llama3"

    @patch("docfinder.web.app._save_settings")
    @patch("docfinder.web.app.load_settings")
    def test_embedding_change_resets_embedder(
        self, mock_load: MagicMock, mock_save: MagicMock
    ) -> None:
        mock_load.return_value = {"hotkey": "<alt>+d", "embedding_backend": "local"}
        with patch("docfinder.web.app._reset_embedder") as mock_reset:
            response = client.post("/settings", json={"embedding_backend": "ollama"})
        assert response.status_code == 200
        mock_reset.assert_called_once()

    @patch("docfinder.web.app._save_settings")
    @patch("docfinder.web.app.load_settings")
    def test_unchanged_settings_do_not_reset_embedder(
        self, mock_load: MagicMock, mock_save: MagicMock
    ) -> None:
        mock_load.return_value = {"hotkey": "<alt>+d", "embedding_backend": "local"}
        with patch("docfinder.web.app._reset_embedder") as mock_reset:
            response = client.post("/settings", json={"hotkey": "<cmd>+k"})
        assert response.status_code == 200
        mock_reset.assert_not_called()


class TestSearchIndexCompat:
    """Tests for 409 Reindex-required behavior on /search (dimension mismatch)."""

    @patch("docfinder.web.app._get_embedder")
    def test_search_409_on_dimension_mismatch(self, mock_get_embedder: MagicMock, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        store = SQLiteVectorStore(db_path, dimension=384)
        store.close()

        embedder = MagicMock()
        embedder.model_name = "new-model"
        embedder.dimension = 384
        mock_get_embedder.return_value = embedder
        with patch("docfinder.web.app.Searcher") as mock_searcher_class:
            mock_searcher_class.return_value.search.side_effect = ValueError(
                "Index vectors have 768 dimensions but the current embedding model "
                "produces 384. The index was built with a different embedding model "
                "— re-index your documents."
            )
            response = client.post("/search", json={"query": "test", "db": str(db_path)})
        assert response.status_code == 409
        assert "re-index" in response.json()["detail"]

    @patch("docfinder.web.app._get_embedder")
    def test_search_no_meta_passes(self, mock_get_embedder: MagicMock, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        SQLiteVectorStore(db_path, dimension=384).close()

        embedder = MagicMock()
        embedder.model_name = "m"
        embedder.dimension = 384
        mock_get_embedder.return_value = embedder
        with patch("docfinder.web.app.Searcher") as mock_searcher_class:
            mock_searcher_class.return_value.search.return_value = []
            response = client.post("/search", json={"query": "test", "db": str(db_path)})
        assert response.status_code == 200
