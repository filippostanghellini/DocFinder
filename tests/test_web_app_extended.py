"""Extended tests for web app endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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
