"""Extended tests for web app endpoints."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from docfinder.index.storage import SQLiteVectorStore
from docfinder.models import ChunkRecord, DocumentMetadata
from docfinder.ollama import OllamaError
from docfinder.web.app import app

client = TestClient(app)


def _post_index_and_wait(payload: dict):
    with patch("docfinder.web.app._preload_embedder"):
        with TestClient(app) as test_client:
            response = test_client.post("/index", json=payload)
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    job = test_client.get(f"/index/status/{job_id}").json()
                    if job["status"] != "running":
                        assert job["status"] == "complete", job
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError(f"Index job {job_id} did not finish")
            return response


class TestSettingsEndpoints:
    """Tests for GET/POST /settings endpoints."""

    @patch("docfinder.web.app.load_settings")
    def test_get_settings(self, mock_load_settings: MagicMock) -> None:
        """Returns current settings."""
        mock_load_settings.return_value = {
            "hotkey": "<cmd>+space",
            "hotkey_enabled": True,
            "ollama_api_key": "secret-token",
        }
        response = client.get("/settings")
        assert response.status_code == 200
        assert response.json()["hotkey"] == "<cmd>+space"
        assert response.json()["hotkey_enabled"] is True
        assert response.json()["has_ollama_api_key"] is True
        assert "ollama_api_key" not in response.json()
        assert "secret-token" not in response.text

    @patch("docfinder.web.app.load_settings")
    @patch("docfinder.web.app._save_settings")
    def test_update_settings(self, mock_save: MagicMock, mock_load: MagicMock) -> None:
        """Updates and returns settings."""
        mock_load.return_value = {
            "hotkey": "<alt>+d",
            "hotkey_enabled": True,
            "ollama_api_key": "secret-token",
        }
        response = client.post("/settings", json={"hotkey": "<cmd>+f"})
        assert response.status_code == 200
        assert response.json()["hotkey"] == "<cmd>+f"
        assert response.json()["has_ollama_api_key"] is True
        assert "ollama_api_key" not in response.json()
        assert "secret-token" not in response.text
        assert mock_save.call_args.args[0]["ollama_api_key"] == "secret-token"

    def test_cors_does_not_allow_arbitrary_origins(self) -> None:
        response = client.options(
            "/settings",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in response.headers


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
    """Tests for POST /api/ollama/models endpoint."""

    @patch("docfinder.web.app.list_ollama_models")
    def test_connected(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [{"name": "llama3", "size_bytes": 123}]
        response = client.post("/api/ollama/models", json={"url": "http://x", "api_key": ""})
        assert response.status_code == 200
        assert response.json() == {
            "connected": True,
            "models": [{"name": "llama3", "size_bytes": 123}],
            "error": None,
        }

    @patch("docfinder.web.app.list_ollama_models")
    def test_unreachable_never_raises(self, mock_list: MagicMock) -> None:
        mock_list.side_effect = OllamaError("Ollama server unreachable at http://x: boom")
        response = client.post("/api/ollama/models", json={"url": "http://x", "api_key": ""})
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert "unreachable" in data["error"]

    def test_no_url_configured(self) -> None:
        response = client.post("/api/ollama/models", json={})
        assert response.status_code == 200
        assert response.json()["connected"] is False

    @patch("docfinder.web.app.list_ollama_models")
    @patch("docfinder.web.app.load_settings")
    def test_saved_key_only_used_for_saved_server(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = {
            "ollama_url": "https://ollama.example",
            "ollama_api_key": "saved-secret",
        }
        mock_list.return_value = []

        client.post("/api/ollama/models", json={"url": "https://attacker.example"})
        mock_list.assert_called_once_with("https://attacker.example", api_key="", timeout=5)

        mock_list.reset_mock()
        client.post("/api/ollama/models", json={"url": "https://ollama.example"})
        mock_list.assert_called_once_with(
            "https://ollama.example", api_key="saved-secret", timeout=5
        )


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

    def test_missing_source_aborts_without_replacing_index(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        present = tmp_path / "present"
        present.mkdir()
        missing = tmp_path / "temporarily-unmounted"
        store = SQLiteVectorStore(db_path, dimension=3)
        doc = DocumentMetadata(
            path=tmp_path / "existing.pdf", title="existing", sha256="old", mtime=1, size=1
        )
        store.upsert_document(
            doc,
            [ChunkRecord(document_path=doc.path, index=0, text="kept", metadata={})],
            np.ones((1, 3), dtype="float32"),
        )
        store.set_meta(
            "source_paths",
            json.dumps([str(present), {"path": str(missing), "privacy": True}]),
        )
        store.close()

        with patch("docfinder.web.app._run_index_job") as mock_run:
            response = client.post(f"/index/reindex?db={db_path}")

        assert response.status_code == 400
        assert "missing" in response.json()["detail"].lower()
        mock_run.assert_not_called()
        store = SQLiteVectorStore(db_path, dimension=3)
        try:
            assert [doc["title"] for doc in store.list_documents()] == ["existing"]
        finally:
            store.close()

    def test_document_mutations_wait_while_reindex_holds_write_lock(self, tmp_path) -> None:
        import docfinder.web.app as web_app

        db_path = tmp_path / "test.db"
        SQLiteVectorStore(db_path, dimension=3).close()
        started = threading.Event()
        finished = threading.Event()

        def write() -> None:
            started.set()
            web_app._write_store(db_path, 3, lambda store: store.clear_all())
            finished.set()

        with web_app._db_write_lock:
            worker = threading.Thread(target=write)
            worker.start()
            assert started.wait(1)
            assert not finished.wait(0.05)

        assert finished.wait(1)
        worker.join()

    def test_starts_index_job(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        store = SQLiteVectorStore(db_path, dimension=0)
        store.set_meta("source_paths", json.dumps([str(tmp_path)]))
        store.close()

        with patch("docfinder.web.app._get_embedder", return_value=MagicMock(dimension=384)):
            with patch("docfinder.web.app._run_index_job") as mock_run:
                mock_run.return_value = {
                    "inserted": 1,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "processed_files": [],
                }
                response = client.post(f"/index/reindex?db={db_path}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "job_id" in response.json()

    def test_rebuild_copies_completed_stage_to_live_index(self, tmp_path) -> None:
        import docfinder.web.app as web_app

        db_path = tmp_path / "test.db"
        source = tmp_path / "source"
        source.mkdir()
        store = SQLiteVectorStore(db_path, dimension=384)
        old_doc = DocumentMetadata(
            path=tmp_path / "old.pdf", title="old", sha256="old", mtime=1, size=1
        )
        store.upsert_document(
            old_doc,
            [ChunkRecord(document_path=old_doc.path, index=0, text="old", metadata={})],
            np.ones((1, 384), dtype="float32"),
        )
        store.set_meta("source_paths", json.dumps([str(source)]))
        store.close()

        def index_stage(paths, config, staging_db, job, exclusions, privacy, *, lock_db):
            assert web_app._db_write_lock.locked()
            assert lock_db is False
            staged = SQLiteVectorStore(staging_db, dimension=384)
            new_doc = DocumentMetadata(
                path=tmp_path / "new.pdf", title="new", sha256="new", mtime=2, size=2
            )
            staged.upsert_document(
                new_doc,
                [ChunkRecord(document_path=new_doc.path, index=0, text="new", metadata={})],
                np.ones((1, 384), dtype="float32"),
            )
            staged.close()
            return {
                "inserted": 1,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "processed_files": [str(new_doc.path)],
            }

        job = {"id": "test-rebuild", "processed": 0, "total": 0, "current_file": ""}
        with patch("docfinder.web.app._get_embedder", return_value=MagicMock(dimension=384)):
            with patch("docfinder.web.app._run_index_job", side_effect=index_stage):
                result = web_app._rebuild_index(db_path, job)

        assert result["inserted"] == 1
        store = SQLiteVectorStore(db_path, dimension=384)
        try:
            assert [doc["title"] for doc in store.list_documents()] == ["new"]
            assert json.loads(store.get_meta("source_paths")) == [
                {"path": str(source), "privacy": False, "exclude_paths": []}
            ]
        finally:
            store.close()

    def test_rebuild_keeps_original_if_source_disappears_midrun(self, tmp_path) -> None:
        import docfinder.web.app as web_app

        db_path = tmp_path / "test.db"
        source = tmp_path / "source"
        source.mkdir()
        store = SQLiteVectorStore(db_path, dimension=3)
        old_doc = DocumentMetadata(
            path=tmp_path / "old.pdf", title="old", sha256="old", mtime=1, size=1
        )
        store.upsert_document(
            old_doc,
            [ChunkRecord(document_path=old_doc.path, index=0, text="old", metadata={})],
            np.ones((1, 3), dtype="float32"),
        )
        store.set_meta("source_paths", json.dumps([str(source)]))
        store.close()

        def remove_source(*args, **kwargs):
            source.rmdir()
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "processed_files": [],
            }

        with patch("docfinder.web.app._get_embedder", return_value=MagicMock(dimension=3)):
            with patch("docfinder.web.app._run_index_job", side_effect=remove_source):
                try:
                    web_app._rebuild_index(db_path, {"id": "vanished-source"})
                except Exception as exc:
                    assert "reconnect missing source paths" in str(exc)
                else:
                    raise AssertionError("Reindex should abort when a source disappears")

        store = SQLiteVectorStore(db_path, dimension=3)
        try:
            assert [doc["title"] for doc in store.list_documents()] == ["old"]
        finally:
            store.close()

    def test_failed_reindex_keeps_original_index(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        source = tmp_path / "source"
        source.mkdir()
        store = SQLiteVectorStore(db_path, dimension=3)
        doc = DocumentMetadata(
            path=tmp_path / "existing.pdf", title="existing", sha256="old", mtime=1, size=1
        )
        store.upsert_document(
            doc,
            [ChunkRecord(document_path=doc.path, index=0, text="kept", metadata={})],
            np.ones((1, 3), dtype="float32"),
        )
        store.set_meta("source_paths", json.dumps([{"path": str(source), "privacy": False}]))
        store.close()

        with patch("docfinder.web.app._get_embedder", return_value=MagicMock(dimension=3)):
            with patch("docfinder.web.app._run_index_job") as mock_run:
                mock_run.return_value = {
                    "inserted": 0,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 1,
                    "processed_files": [str(source / "bad.pdf")],
                }
                response = client.post(f"/index/reindex?db={db_path}")

        assert response.status_code == 200
        store = SQLiteVectorStore(db_path, dimension=3)
        try:
            assert [doc["title"] for doc in store.list_documents()] == ["existing"]
        finally:
            store.close()


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

    @patch("docfinder.web.app._schedule_rag_load")
    @patch("docfinder.web.app._save_settings")
    @patch("docfinder.web.app.load_settings")
    def test_changed_chat_model_reloads_when_rag_enabled(
        self, mock_load: MagicMock, mock_save: MagicMock, mock_schedule: MagicMock
    ) -> None:
        mock_load.return_value = {
            "rag_enabled": True,
            "llm_backend": "local",
            "llm_model": "",
        }

        response = client.post("/settings", json={"llm_backend": "ollama", "llm_model": "llama3"})

        assert response.status_code == 200
        mock_schedule.assert_called_once_with()


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


class TestPrivacyMode:
    """Tests for the 100% privacy feature."""

    @patch("docfinder.web.app._get_embedder")
    def test_index_privacy_with_remote_ollama_rejected(
        self, mock_get_embedder: MagicMock, tmp_path
    ) -> None:
        from docfinder.ollama import OllamaEmbedder

        mock_get_embedder.return_value = OllamaEmbedder("http://vps.example.com:11434", "m")
        response = client.post("/index", json={"paths": [str(tmp_path)], "privacy": True})
        assert response.status_code == 400
        assert "privacy" in response.json()["detail"].lower()

    def test_index_privacy_with_local_embedder_accepted(self, tmp_path) -> None:
        with patch("docfinder.web.app._get_embedder") as mock_get:
            mock_get.return_value = MagicMock(dimension=384)
            with patch("docfinder.web.app._validate_paths") as mock_validate:
                mock_validate.return_value = [tmp_path]
                with patch("docfinder.web.app._run_index_job") as mock_run:
                    mock_run.return_value = {
                        "inserted": 0,
                        "updated": 0,
                        "skipped": 0,
                        "failed": 0,
                        "processed_files": [],
                    }
                    response = _post_index_and_wait(
                        {
                            "paths": [str(tmp_path)],
                            "privacy": True,
                            "db": str(tmp_path / "index.db"),
                        }
                    )
        assert response.status_code == 200
        assert mock_run.call_args[0][5] is True  # privacy flag reaches the job

    def test_index_privacy_with_localhost_ollama_accepted(self, tmp_path) -> None:
        from docfinder.ollama import OllamaEmbedder

        with patch("docfinder.web.app._get_embedder") as mock_get:
            mock_get.return_value = OllamaEmbedder("http://127.0.0.1:11434", "m")
            with patch("docfinder.web.app._validate_paths") as mock_validate:
                mock_validate.return_value = [tmp_path]
                with patch("docfinder.web.app._run_index_job") as mock_run:
                    mock_run.return_value = {
                        "inserted": 0,
                        "updated": 0,
                        "skipped": 0,
                        "failed": 0,
                        "processed_files": [],
                    }
                    response = _post_index_and_wait(
                        {
                            "paths": [str(tmp_path)],
                            "privacy": True,
                            "db": str(tmp_path / "index.db"),
                        }
                    )
        assert response.status_code == 200

    def test_chat_privacy_doc_with_remote_ollama_forbidden(self, tmp_path) -> None:
        from docfinder.ollama import OllamaLLM

        db_path = tmp_path / "test.db"
        store = SQLiteVectorStore(db_path, dimension=384)
        doc = DocumentMetadata(
            path=Path(r"C:\Users\tester\private.pdf"),
            title="P",
            sha256="x",
            mtime=1.0,
            size=10,
        )
        store.upsert_document(
            doc,
            [ChunkRecord(document_path=doc.path, index=0, text="t", metadata={})],
            np.random.rand(1, 384).astype("float32"),
            privacy=True,
        )
        store.close()

        with patch("docfinder.web.app._get_embedder") as mock_get:
            mock_get.return_value = MagicMock(dimension=384)
            import docfinder.web.app as web_app

            original = web_app._rag_llm
            web_app._rag_llm = OllamaLLM("http://vps.example.com:11434", "m")
            try:
                response = client.post(
                    "/rag/chat",
                    json={
                        "question": "test?",
                        "document_path": "C:/Users/tester/private.pdf",
                        "chunk_index": 0,
                        "db": str(db_path),
                    },
                )
                assert response.status_code == 403
                assert "privacy" in response.json()["detail"].lower()
            finally:
                web_app._rag_llm = original

    def test_chat_privacy_doc_with_localhost_ollama_allowed(self, tmp_path) -> None:
        from docfinder.ollama import OllamaLLM

        db_path = tmp_path / "test.db"
        store = SQLiteVectorStore(db_path, dimension=384)
        doc = DocumentMetadata(
            path=Path(r"C:\Users\tester\private.pdf"),
            title="P",
            sha256="x",
            mtime=1.0,
            size=10,
        )
        store.upsert_document(
            doc,
            [ChunkRecord(document_path=doc.path, index=0, text="t", metadata={})],
            np.random.rand(1, 384).astype("float32"),
            privacy=True,
        )
        store.close()

        fake_llm = OllamaLLM("http://127.0.0.1:11434", "m")
        with patch("docfinder.web.app._get_embedder") as mock_get:
            mock_get.return_value = MagicMock(dimension=384)
            import docfinder.web.app as web_app

            original = web_app._rag_llm
            web_app._rag_llm = fake_llm
            try:
                with patch.object(OllamaLLM, "chat", return_value="answer"):
                    response = client.post(
                        "/rag/chat",
                        json={
                            "question": "test?",
                            "document_path": "C:/Users/tester/private.pdf",
                            "chunk_index": 0,
                            "db": str(db_path),
                        },
                    )
                assert response.status_code == 200
                assert response.json()["answer"] == "answer"
            finally:
                web_app._rag_llm = original

    def test_reindex_preserves_privacy_paths(self, tmp_path) -> None:
        db_path = tmp_path / "test.db"
        store = SQLiteVectorStore(db_path, dimension=384)
        store.set_meta(
            "source_paths",
            json.dumps(
                [
                    {
                        "path": str(tmp_path / "private"),
                        "privacy": True,
                        "exclude_paths": [str(tmp_path / "private" / "skip.pdf")],
                    },
                    {"path": str(tmp_path / "open"), "privacy": False},
                ]
            ),
        )
        store.close()
        (tmp_path / "private").mkdir()
        (tmp_path / "open").mkdir()

        import docfinder.web.app as web_app

        with patch("docfinder.web.app._run_index_job") as mock_run:
            mock_run.return_value = {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "processed_files": [],
            }

            def assert_locked(*args, **kwargs):
                assert web_app._db_write_lock.locked()
                return mock_run.return_value

            mock_run.side_effect = assert_locked
            with patch("docfinder.web.app._get_embedder") as mock_get:
                mock_get.return_value = MagicMock(dimension=384)
                result = web_app._rebuild_index(
                    db_path,
                    {"id": "privacy-reindex", "processed": 0, "total": 0, "current_file": ""},
                )

        assert result["failed"] == 0
        privacy_flags = [call.args[5] for call in mock_run.call_args_list]
        assert sorted(privacy_flags, key=str) == sorted([True, False], key=str)
        private_call = next(call for call in mock_run.call_args_list if call.args[5])
        assert private_call.args[4] == frozenset({str(tmp_path / "private" / "skip.pdf")})

    def test_index_manifest_accumulates_roots_and_exclusions(self, tmp_path) -> None:
        db_path = tmp_path / "manifest.db"
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        for path, excluded, privacy in (
            (first, first / "skip.pdf", False),
            (second, second / "private.pdf", True),
        ):
            with patch("docfinder.web.app._get_embedder", return_value=MagicMock(dimension=3)):
                with patch("docfinder.web.app._validate_paths", return_value=[path]):
                    with patch("docfinder.web.app._run_index_job", return_value={"failed": 0}):
                        response = _post_index_and_wait(
                            {
                                "paths": [str(path)],
                                "db": str(db_path),
                                "exclude_paths": [str(excluded)],
                                "privacy": privacy,
                            }
                        )
            assert response.status_code == 200

        store = SQLiteVectorStore(db_path, dimension=0)
        try:
            entries = json.loads(store.get_meta("source_paths"))
        finally:
            store.close()
        assert entries == [
            {
                "path": str(first),
                "privacy": False,
                "exclude_paths": [str(first / "skip.pdf")],
            },
            {
                "path": str(second),
                "privacy": True,
                "exclude_paths": [str(second / "private.pdf")],
            },
        ]
