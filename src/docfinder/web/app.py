"""FastAPI application backing the DocFinder web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from docfinder.config import AppConfig
from docfinder.embedding.encoder import (
    EmbeddingConfig,
    EmbeddingModel,
    get_runtime_environment_info,
)
from docfinder.index.indexer import Indexer
from docfinder.index.reranker import Reranker
from docfinder.index.search import Searcher, SearchResult
from docfinder.index.storage import SQLiteVectorStore
from docfinder.ollama import (
    OllamaEmbedder,
    OllamaError,
    OllamaLLM,
    is_local_url,
    list_ollama_models,
)
from docfinder.settings import load_settings
from docfinder.settings import save_settings as _save_settings
from docfinder.web.frontend import router as frontend_router

LOGGER = logging.getLogger(__name__)

# ── Singleton EmbeddingModel ─────────────────────────────────────────────────
_embedder: EmbeddingModel | OllamaEmbedder | None = None
_embedder_lock = threading.Lock()


def _build_embedder_from_settings() -> EmbeddingModel | OllamaEmbedder:
    """Build the embedder from persisted settings (Ollama or local SentenceTransformer)."""
    settings = load_settings()
    if settings.get("embedding_backend") == "ollama" and settings.get("ollama_url"):
        return OllamaEmbedder(
            settings["ollama_url"],
            settings.get("embedding_model") or "",
            api_key=settings.get("ollama_api_key") or "",
        )
    config = AppConfig()
    model_name = settings.get("embedding_model") or config.model_name
    return EmbeddingModel(EmbeddingConfig(model_name=model_name))


def _get_embedder() -> EmbeddingModel | OllamaEmbedder:
    """Return a cached EmbeddingModel, creating it on first call."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = _build_embedder_from_settings()
    return _embedder


def _preload_embedder() -> None:
    """Warm the embedding model in the background.

    Server startup (and the desktop window) must never block on a slow model
    download or load; a failed preload is retried lazily on first request.
    """
    try:
        _get_embedder()
    except Exception:
        LOGGER.exception("Embedding model preload failed — will retry on first use")


def _reset_embedder() -> None:
    """Drop the cached embedder so the next call rebuilds it from settings."""
    global _embedder
    with _embedder_lock:
        _embedder = None


# ── Singleton Reranker ────────────────────────────────────────────────────────
_reranker: Reranker | None = None
_reranker_lock = threading.Lock()


def _get_reranker() -> Reranker:
    """Return a cached Reranker, creating it on first call (lazy model load)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker


# ── Async indexing job registry ───────────────────────────────────────────────
_index_jobs: dict[str, dict] = {}
# ponytail: one in-process write lock; per-DB locks if multi-DB throughput matters.
_db_write_lock = threading.Lock()

# ── GUI callback registry (set by the desktop GUI layer, not used in web mode) ─
_spotlight_hide_callback: object = None  # callable | None
_is_gui_mode: bool = False


def register_spotlight_hide_callback(callback: object) -> None:
    """Register a callable that hides the spotlight panel (called by gui.py)."""
    global _spotlight_hide_callback
    _spotlight_hide_callback = callback


def set_gui_mode(enabled: bool = True) -> None:
    """Mark the app as running inside the desktop GUI (pywebview)."""
    global _is_gui_mode
    _is_gui_mode = enabled


def _notify_indexing_done(result: dict[str, Any] | None, *, error: str | None = None) -> None:
    """Send a native notification when indexing completes (GUI mode only)."""
    if not _is_gui_mode:
        return

    from docfinder.utils.notify import send_notification

    if error:
        send_notification("DocFinder", f"Indexing failed: {error}")
    elif result:
        inserted = result.get("inserted", 0)
        updated = result.get("updated", 0)
        skipped = result.get("skipped", 0)
        total = inserted + updated + skipped
        send_notification("DocFinder", f"Indexing complete: {total} documents processed.")


def _preload_reranker() -> None:
    """Pre-load the reranker model (singleton + ensure weights are downloaded)."""
    reranker = _get_reranker()
    reranker._ensure_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Pre-load the embedder in the background so the server starts (and the
    # desktop window opens) without waiting for the model download/load.
    threading.Thread(target=_preload_embedder, daemon=True).start()
    yield


app = FastAPI(title="DocFinder Web", version="2.3.0", lifespan=lifespan)
app.include_router(frontend_router)


class SearchPayload(BaseModel):
    query: str
    db: Path | None = None
    top_k: int = 10
    folders: List[str] = []


class OpenRequest(BaseModel):
    path: Path


class DeleteDocumentRequest(BaseModel):
    doc_id: int | None = None
    path: str | None = None


class IndexPayload(BaseModel):
    paths: List[str]
    db: str | None = None
    model: str | None = None
    chunk_chars: int | None = None
    overlap: int | None = None
    exclude_paths: List[str] = []
    privacy: bool = False


class RAGPayload(BaseModel):
    question: str
    document_path: str
    chunk_index: int
    db: Path | None = None


class SettingsPayload(BaseModel):
    hotkey: str | None = None
    hotkey_enabled: bool | None = None
    rag_enabled: bool | None = None
    rag_model: str | None = None
    embedding_backend: str | None = None
    embedding_model: str | None = None
    ollama_url: str | None = None
    ollama_api_key: str | None = None
    llm_backend: str | None = None
    llm_model: str | None = None


class OllamaModelsPayload(BaseModel):
    url: str | None = None
    api_key: str | None = None


_EMBEDDER_RESET_KEYS = ("embedding_backend", "embedding_model", "ollama_url", "ollama_api_key")
_LLM_RESET_KEYS = ("llm_backend", "llm_model", "rag_model", "ollama_url", "ollama_api_key")


def _resolve_db_path(db: Path | None) -> Path:
    config = AppConfig(db_path=db if db is not None else AppConfig().db_path)
    return config.resolve_db_path(Path.cwd())


def _ensure_db_parent(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _write_store(db_path: Path, dimension: int, operation: Callable) -> Any:
    with _db_write_lock:
        store = SQLiteVectorStore(db_path, dimension=dimension)
        try:
            return operation(store)
        finally:
            store.close()


@app.post("/search")
async def search_documents(payload: SearchPayload) -> dict[str, List[SearchResult]]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    top_k = max(1, min(payload.top_k, 50))

    resolved_db = _resolve_db_path(payload.db)
    if not resolved_db.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Database not found at {resolved_db}. "
            "Please index some documents first using the 'Index folder or PDF' section above.",
        )

    embedder = _get_embedder()
    reranker = _get_reranker()
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    searcher = Searcher(embedder, store, reranker=reranker)
    folders = [f.strip() for f in payload.folders if f and f.strip()]
    try:
        results = searcher.search(query, top_k=top_k, folders=folders if folders else None)
    except ValueError as exc:
        # e.g. index built with a different embedding model — tell the user
        # instead of leaking a raw 500.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()
    return {"results": results}


@app.get("/search/folders")
async def search_folders(db: Path | None = None) -> dict[str, Any]:
    """Return currently indexed directories for search-time filtering."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        return {"folders": []}

    embedder = _get_embedder()
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        folders = store.list_indexed_directories()
    finally:
        store.close()

    return {"folders": folders}


# ── RAG singleton + download progress ─────────────────────────────────────
_rag_llm: Any = None
_rag_llm_lock = threading.Lock()
_rag_download: dict[str, Any] = {
    "status": "idle",  # idle | downloading | loading | ready | error
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "error": None,
}


def _load_rag_llm(model_name: str | None = None) -> None:
    """Download (if needed) and load the RAG LLM.  Updates _rag_download state.

    Uses a remote Ollama LLM when configured in settings, otherwise the
    local llama.cpp GGUF pipeline.
    """
    global _rag_llm
    settings = load_settings()
    ollama_llm = (
        settings.get("llm_backend") == "ollama"
        and bool(settings.get("llm_model"))
        and bool(settings.get("ollama_url"))
    )
    if ollama_llm:
        _rag_download["status"] = "loading"
        _rag_download["error"] = None
        try:
            list_ollama_models(settings["ollama_url"], api_key=settings.get("ollama_api_key") or "")
        except OllamaError as exc:
            _rag_download["status"] = "error"
            _rag_download["error"] = str(exc)
            return
        _rag_llm = OllamaLLM(
            settings["ollama_url"],
            settings["llm_model"],
            api_key=settings.get("ollama_api_key") or "",
        )
        _rag_download["status"] = "ready"
        return

    from docfinder.rag.llm import _DEFAULT_MODELS_DIR, MODEL_TIERS, LocalLLM, ModelSpec

    # Pick the requested model or auto-select
    spec: ModelSpec | None = None
    if model_name is None:
        model_name = settings.get("rag_model") or None
    if model_name:
        for t in MODEL_TIERS:
            if t.name == model_name:
                spec = t
                break
    if spec is None:
        from docfinder.rag.llm import select_model

        spec = select_model()

    dest_dir = _DEFAULT_MODELS_DIR
    local_path = dest_dir / spec.filename

    if not local_path.exists():
        _rag_download["status"] = "downloading"
        _rag_download["downloaded_bytes"] = 0
        _rag_download["total_bytes"] = 0
        _rag_download["error"] = None

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils.tqdm import tqdm as HfTqdm

        # Monkey-patch the HF tqdm class to capture download progress
        _orig_init = HfTqdm.__init__
        _orig_update = HfTqdm.update
        outer = _rag_download

        def _patched_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            if self.total:
                outer["total_bytes"] = int(self.total)

        def _patched_update(self, n=1):
            _orig_update(self, n)
            outer["downloaded_bytes"] = int(self.n)

        HfTqdm.__init__ = _patched_init
        HfTqdm.update = _patched_update
        try:
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=spec.filename,
                local_dir=str(dest_dir),
            )
        finally:
            HfTqdm.__init__ = _orig_init
            HfTqdm.update = _orig_update

    _rag_download["status"] = "loading"
    _rag_llm = LocalLLM(local_path, n_ctx=spec.ctx_size)
    _rag_download["status"] = "ready"


def _schedule_rag_load(model_name: str | None = None) -> None:
    """Load the selected chat model in the background."""
    _rag_download["status"] = "downloading"
    _rag_download["error"] = None

    async def _run() -> None:
        try:
            await asyncio.to_thread(_load_rag_llm, model_name)
        except Exception as exc:
            LOGGER.exception("RAG model download/load failed: %s", exc)
            _rag_download["status"] = "error"
            _rag_download["error"] = str(exc)

    asyncio.create_task(_run())


@app.get("/rag/models")
async def rag_models() -> dict:
    """Return available model tiers with a recommended flag."""
    from docfinder.rag.llm import _DEFAULT_MODELS_DIR, MODEL_TIERS, select_model

    recommended = select_model()
    total_ram = await asyncio.to_thread(_get_total_ram_for_rag)

    models = []
    for spec in MODEL_TIERS:
        local_path = _DEFAULT_MODELS_DIR / spec.filename
        models.append(
            {
                "name": spec.name,
                "filename": spec.filename,
                "ram_min_mb": spec.ram_min_mb,
                "recommended": spec.name == recommended.name,
                "downloaded": local_path.exists(),
                "size_label": _format_size_label(spec),
            }
        )
    return {"models": models, "total_ram_mb": total_ram}


def _get_total_ram_for_rag() -> int:
    from docfinder.rag.llm import _get_total_ram_mb

    return _get_total_ram_mb()


def _format_size_label(spec) -> str:
    """Return a human-readable approximate download size."""
    sizes = {
        "Qwen3.5-9B": "~5.7 GB",
        "Qwen3.5-4B": "~2.7 GB",
        "Qwen3.5-2B": "~1.3 GB",
    }
    return sizes.get(spec.name, "unknown")


@app.post("/rag/download")
async def rag_download(model_name: str | None = None) -> dict:
    """Start downloading and loading the RAG model in background."""
    if _rag_download["status"] in ("downloading", "loading"):
        return {"status": "already_running"}

    # Read user preference from settings
    settings = load_settings()
    chosen = model_name or settings.get("rag_model")
    _schedule_rag_load(chosen)
    return {"status": "started"}


@app.get("/rag/download/status")
async def rag_download_status() -> dict:
    """Poll download / load progress."""
    return dict(_rag_download)


@app.post("/rag/chat")
async def rag_chat(payload: RAGPayload) -> dict:
    """Answer a question using RAG over the context window of a specific chunk."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if _rag_llm is None:
        raise HTTPException(
            status_code=503,
            detail="RAG model not loaded. Enable AI Chat in Settings and download a model first.",
        )

    resolved_db = _resolve_db_path(payload.db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        # Look up document_id
        row = store.connection.execute(
            "SELECT id, privacy FROM documents WHERE path = ?", (payload.document_path,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found in index")
        doc_id = row["id"]

        if (
            row["privacy"]
            and isinstance(_rag_llm, OllamaLLM)
            and not is_local_url(_rag_llm.base_url)
        ):
            raise HTTPException(
                status_code=403,
                detail="This document was indexed in 100% privacy mode — "
                "chat requires a local or localhost LLM",
            )

        # Get context: try page-based first, fall back to fixed window
        import json as _json

        max_tokens_for_context = _rag_llm.n_ctx - 800  # reserve ~800 tokens for prompt overhead
        max_chars = max(max_tokens_for_context * 4, 1000)  # ~4 chars per token

        # Find the page of the clicked chunk
        chunk_row = store.connection.execute(
            "SELECT metadata FROM chunks WHERE document_id = ? AND chunk_index = ?",
            (doc_id, payload.chunk_index),
        ).fetchone()
        chunk_meta = (
            _json.loads(chunk_row["metadata"]) if chunk_row and chunk_row["metadata"] else {}
        )
        center_page = chunk_meta.get("page") if chunk_row else None

        if center_page is not None:
            # Page-based context: same page + expand to adjacent pages
            context_chunks = store.get_context_by_page(doc_id, center_page, max_chars=max_chars)
        else:
            # Fallback for old indexes without page metadata
            context_chunks = store.get_context_window(doc_id, payload.chunk_index, window_size=10)

        if not context_chunks:
            raise HTTPException(status_code=404, detail="No chunks found for this document")

        # Assemble context text respecting the token budget
        parts = []
        total = 0
        for c in context_chunks:
            text = c["text"]
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(text[:remaining] + " [...]")
                break
            parts.append(text)
            total += len(text)
        context_text = "\n\n".join(parts)

        # Generate answer
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided document "
            "context. Use ONLY the information from the context below. "
            "If the context does not contain enough information, say so clearly. "
            "Answer in the same language as the user's question."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        try:
            answer = await asyncio.to_thread(
                _rag_llm.chat, messages, max_tokens=1024, temperature=0.2
            )
        except Exception as exc:
            LOGGER.exception("RAG LLM chat failed")
            raise HTTPException(
                status_code=502,
                detail=f"RAG model failed to generate a response: {exc}",
            )
    finally:
        store.close()

    return {"answer": answer, "context_chunks_used": len(context_chunks)}


@app.post("/open")
async def open_document(payload: OpenRequest) -> dict[str, str]:
    path = payload.path.expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        if os.name == "posix":  # macOS/Linux
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
        else:
            normalized_path = os.path.normpath(str(path))
            if not Path(normalized_path).is_absolute():
                normalized_path = os.path.abspath(normalized_path)
            os.startfile(normalized_path)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Unable to open %s: %s", path, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok"}


@app.post("/gui/spotlight/hide", include_in_schema=False)
async def spotlight_hide() -> dict[str, str]:
    """Signal the native SpotlightPanel to hide (called by spotlight.html JS)."""
    cb = _spotlight_hide_callback
    if cb is not None:
        await asyncio.to_thread(cb)  # type: ignore[arg-type]
    return {"status": "ok"}


@app.get("/documents")
async def list_documents(db: Path | None = None) -> dict[str, Any]:
    """List all indexed documents in the database."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        return {
            "documents": [],
            "stats": {"document_count": 0, "chunk_count": 0, "total_size_bytes": 0},
        }

    embedder = _get_embedder()
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        documents = store.list_documents()
        stats = store.get_stats()
    finally:
        store.close()

    return {"documents": documents, "stats": stats}


@app.delete("/documents/cleanup")
async def cleanup_missing_files(db: Path | None = None) -> dict[str, Any]:
    """Remove documents whose files no longer exist on disk."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()
    removed_count = await asyncio.to_thread(
        _write_store,
        resolved_db,
        embedder.dimension,
        lambda store: store.remove_missing_files(),
    )

    return {"status": "ok", "removed_count": removed_count}


@app.delete("/documents/delete-all")
async def delete_all_documents(db: Path | None = None) -> dict[str, Any]:
    """Remove every indexed document and chunk (used before reindexing)."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()
    removed = await asyncio.to_thread(
        _write_store, resolved_db, embedder.dimension, lambda store: store.clear_all()
    )

    return {"status": "ok", "removed": removed}


@app.delete("/documents/{doc_id}")
async def delete_document_by_id(doc_id: int, db: Path | None = None) -> dict[str, Any]:
    """Delete a document by its ID."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()
    deleted = await asyncio.to_thread(
        _write_store, resolved_db, embedder.dimension, lambda store: store.delete_document(doc_id)
    )

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")

    return {"status": "ok", "deleted_id": doc_id}


@app.post("/documents/delete")
async def delete_document(payload: DeleteDocumentRequest, db: Path | None = None) -> dict[str, Any]:
    """Delete a document by ID or path."""
    if payload.doc_id is None and payload.path is None:
        raise HTTPException(status_code=400, detail="Either doc_id or path must be provided")

    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()

    def operation(store: SQLiteVectorStore) -> bool:
        if payload.doc_id is not None:
            return store.delete_document(payload.doc_id)
        return store.delete_document_by_path(payload.path)  # type: ignore[arg-type]

    deleted = await asyncio.to_thread(_write_store, resolved_db, embedder.dimension, operation)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "ok"}


@app.get("/settings")
async def get_settings() -> dict:
    """Return current user settings."""
    return _public_settings(load_settings())


def _public_settings(settings: dict) -> dict:
    public = dict(settings)
    public["has_ollama_api_key"] = bool(public.pop("ollama_api_key", ""))
    return public


@app.post("/api/ollama/models")
async def ollama_models(payload: OllamaModelsPayload) -> dict:
    """List models on an Ollama server. Never raises: returns connected=false on error."""
    settings = load_settings()
    base_url = payload.url or settings.get("ollama_url") or ""
    saved_url = (settings.get("ollama_url") or "").rstrip("/")
    key = payload.api_key
    if key is None:
        key = (settings.get("ollama_api_key") or "") if base_url.rstrip("/") == saved_url else ""
    if not base_url:
        return {"connected": False, "models": [], "error": "No Ollama URL configured"}
    try:
        models = await asyncio.to_thread(list_ollama_models, base_url, api_key=key, timeout=5)
        return {"connected": True, "models": models, "error": None}
    except OllamaError as exc:
        return {"connected": False, "models": [], "error": str(exc)}


@app.post("/index/reindex")
async def reindex_all(db: Path | None = None) -> dict[str, Any]:
    """Rebuild the recorded sources and atomically replace the index on success."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    entries = await asyncio.to_thread(_load_source_manifest_locked, resolved_db)
    _require_available_sources(entries)

    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id": job_id,
        "status": "running",
        "processed": 0,
        "total": 0,
        "current_file": "",
        "stats": None,
        "error": None,
    }
    _index_jobs[job_id] = job

    async def _run() -> None:
        try:
            merged = await asyncio.to_thread(_rebuild_index, resolved_db, job)
            job["status"] = "complete"
            job["stats"] = merged
            _notify_indexing_done(merged)
        except Exception as exc:
            LOGGER.exception("Reindex job %s failed: %s", job_id, exc)
            job["status"] = "error"
            job["error"] = str(exc)
            _notify_indexing_done(None, error=str(exc))

    asyncio.create_task(_run())
    return {"status": "ok", "job_id": job_id}


def _load_source_manifest(db_path: Path) -> list[dict[str, Any]]:
    store = SQLiteVectorStore(db_path, dimension=0)
    try:
        raw = store.get_meta("source_paths")
    finally:
        store.close()
    return _normalize_source_manifest(json.loads(raw) if raw else [])


def _load_source_manifest_locked(db_path: Path) -> list[dict[str, Any]]:
    with _db_write_lock:
        return _load_source_manifest(db_path)


def _require_available_sources(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        raise HTTPException(
            status_code=400, detail="No source paths recorded. Index a folder first."
        )
    missing = [entry["path"] for entry in entries if not Path(entry["path"]).exists()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="Re-index cancelled; reconnect missing source paths first: "
            + ", ".join(missing),
        )
    return entries


def _rebuild_index(resolved_db: Path, job: dict) -> dict[str, Any]:
    embedder = _get_embedder()
    dimension = embedder.dimension
    with _db_write_lock:
        job_id = job["id"]
        staging_db = resolved_db.with_name(f".{resolved_db.name}.{job_id}.reindex")
        _ensure_db_parent(resolved_db)
        conn = sqlite3.connect(resolved_db)
        try:
            # Hold SQLite's cross-process write reservation through rebuild and replacement.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT value FROM meta WHERE key = 'source_paths'").fetchone()
            available = _require_available_sources(
                _normalize_source_manifest(json.loads(row[0]) if row else [])
            )

            stage_store = SQLiteVectorStore(staging_db, dimension=dimension)
            try:
                stage_store.set_meta("source_paths", json.dumps(available))
            finally:
                stage_store.close()

            config = AppConfig(db_path=staging_db)
            merged: dict[str, Any] = {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "failed": 0,
                "processed_files": [],
            }
            for entry in available:
                result = _run_index_job(
                    [Path(entry["path"])],
                    config,
                    staging_db,
                    job,
                    frozenset(entry["exclude_paths"]) or None,
                    bool(entry["privacy"]),
                    lock_db=False,
                )
                for key in ("inserted", "updated", "skipped", "failed"):
                    merged[key] += result[key]
                merged["processed_files"].extend(result["processed_files"])
            if merged["failed"]:
                raise RuntimeError(
                    f"Re-index failed for {merged['failed']} files; the existing index was kept"
                )
            _require_available_sources(available)

            conn.execute("ATTACH DATABASE ? AS rebuilt", (str(staging_db),))
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM meta")
            conn.execute(
                """INSERT INTO documents
                   (id, path, title, sha256, mtime, size, privacy, created_at, updated_at)
                   SELECT id, path, title, sha256, mtime, size, privacy, created_at, updated_at
                   FROM rebuilt.documents"""
            )
            conn.execute(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, text, metadata, embedding, created_at)
                   SELECT id, document_id, chunk_index, text, metadata, embedding, created_at
                   FROM rebuilt.chunks"""
            )
            conn.execute("INSERT INTO meta (key, value) SELECT key, value FROM rebuilt.meta")
            conn.commit()
            return merged
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            for suffix in ("", "-wal", "-shm"):
                try:
                    Path(f"{staging_db}{suffix}").unlink()
                except FileNotFoundError:
                    pass


def _normalize_source_manifest(entries: list) -> list[dict[str, Any]]:
    """Read current and legacy source-path manifests into one stable format."""
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        exclusions = entry.get("exclude_paths", [])
        normalized.append(
            {
                "path": entry["path"],
                "privacy": bool(entry.get("privacy", False)),
                "exclude_paths": [p for p in exclusions if isinstance(p, str)]
                if isinstance(exclusions, list)
                else [],
            }
        )
    return normalized


def _merge_source_manifest(
    existing: list, paths: list[Path], privacy: bool, exclude_paths: list[str]
) -> list[dict[str, Any]]:
    manifest = _normalize_source_manifest(existing)
    by_path = {os.path.normcase(os.path.abspath(entry["path"])): entry for entry in manifest}
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        previous = by_path.get(key, {"path": str(path), "privacy": False, "exclude_paths": []})
        previous["privacy"] = bool(previous["privacy"] or privacy)
        previous["exclude_paths"] = sorted(set(exclude_paths))
        by_path[key] = previous
    return list(by_path.values())


def _save_source_manifest(
    db_path: Path, paths: list[Path], privacy: bool, exclude_paths: list[str]
) -> None:
    with _db_write_lock:
        store = SQLiteVectorStore(db_path, dimension=0)
        try:
            raw = store.get_meta("source_paths")
            previous = json.loads(raw) if raw else []
            store.set_meta(
                "source_paths",
                json.dumps(_merge_source_manifest(previous, paths, privacy, exclude_paths)),
            )
        finally:
            store.close()


@app.post("/settings")
async def update_settings(payload: SettingsPayload) -> dict:
    """Persist updated settings and return the full settings dict."""
    current = load_settings()
    before = {key: current.get(key) for key in _EMBEDDER_RESET_KEYS + _LLM_RESET_KEYS}
    for field in (
        "hotkey",
        "hotkey_enabled",
        "rag_enabled",
        "rag_model",
        "embedding_backend",
        "embedding_model",
        "ollama_url",
        "ollama_api_key",
        "llm_backend",
        "llm_model",
    ):
        value = getattr(payload, field)
        if value is not None:
            current[field] = value
    _save_settings(current)

    after = {key: current.get(key) for key in _EMBEDDER_RESET_KEYS + _LLM_RESET_KEYS}
    if any(before[k] != after[k] for k in _EMBEDDER_RESET_KEYS):
        _reset_embedder()
    llm_changed = any(before[k] != after[k] for k in _LLM_RESET_KEYS)
    if llm_changed:
        global _rag_llm
        _rag_llm = None
        if current.get("rag_enabled"):
            _schedule_rag_load()
    return _public_settings(current)


def _compute_embed_batch_size() -> int:
    """Choose embedding batch size based on available RAM to avoid OOM.

    Note: the Indexer now does per-file adaptive throttling internally.
    This is kept for the initial batch size hint passed to the Indexer.
    """
    from docfinder.utils.memory import compute_embed_batch_size, get_memory_info

    info = get_memory_info()
    batch_size, _ = compute_embed_batch_size(info.get("available_mb"))
    return batch_size


def _run_index_job(
    paths: List[Path],
    config: AppConfig,
    resolved_db: Path,
    job: dict | None = None,
    exclude_paths: frozenset[str] | None = None,
    privacy: bool = False,
    *,
    lock_db: bool = True,
) -> dict[str, Any]:
    embedder = _get_embedder()
    dimension = embedder.dimension

    def _progress(processed: int, total: int, current_file: str) -> None:
        if job is not None:
            job["processed"] = processed
            job["total"] = total
            job["current_file"] = current_file

    with _db_write_lock if lock_db else nullcontext():
        store = SQLiteVectorStore(resolved_db, dimension=dimension)
        # No fixed embed_batch_size — Indexer adapts per-file based on available RAM
        indexer = Indexer(
            embedder,
            store,
            chunk_chars=config.chunk_chars,
            overlap=config.overlap,
            progress_callback=_progress,
            privacy=privacy,
        )
        try:
            stats = indexer.index(paths, exclude_paths=exclude_paths)
        finally:
            store.close()

    return {
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "processed_files": [str(path) for path in stats.processed_files],
    }


def _get_memory_info() -> dict[str, Any]:
    """Return available and total RAM in MB. Delegates to shared utility."""
    from docfinder.utils.memory import get_memory_info

    return get_memory_info()


def _get_runtime_info() -> dict[str, Any]:
    """Return runtime backend/device and indexing strategy info."""
    info = get_runtime_environment_info()
    info["indexing_mode"] = "balanced"
    info["cpu_count"] = os.cpu_count()
    return info


def _validate_paths(paths: List[str]) -> List[Path]:
    """Validate and resolve a list of raw path strings. Raises HTTPException on error."""
    logger = logging.getLogger(__name__)
    safe_base_dir = Path(os.path.realpath(str(Path.home())))
    resolved_paths: List[Path] = []

    for p in paths:
        clean_path = p.strip().replace("\r", "").replace("\n", "")
        if not clean_path:
            continue
        if "\0" in clean_path:
            raise HTTPException(status_code=400, detail="Invalid path: contains null byte")

        try:
            expanded_path = os.path.expanduser(clean_path)
            real_path = os.path.realpath(expanded_path)

            if not os.path.isabs(real_path):
                raise HTTPException(status_code=400, detail="Invalid path: must be absolute")

            safe_base_str = str(safe_base_dir) + os.sep
            real_path_str = real_path + os.sep
            if not real_path_str.startswith(safe_base_str):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: path is outside allowed directory",
                )

            validated_path = Path(real_path)
            if not validated_path.exists():
                raise HTTPException(status_code=404, detail="Path not found: %s" % clean_path)
            if not validated_path.is_dir():
                raise HTTPException(
                    status_code=400, detail="Path must be a directory: %s" % clean_path
                )
            resolved_paths.append(validated_path)

        except (ValueError, OSError) as e:
            logger.error("Invalid path '%s': %s", clean_path, e)
            raise HTTPException(status_code=400, detail="Invalid path: %s" % clean_path)

    return resolved_paths


@app.post("/index")
async def index_documents(payload: IndexPayload) -> dict[str, Any]:
    """Start an indexing job and return its ID immediately for progress polling."""
    if not payload.paths:
        raise HTTPException(status_code=400, detail="No path provided")

    embedder = _get_embedder()
    if (
        payload.privacy
        and isinstance(embedder, OllamaEmbedder)
        and not is_local_url(embedder.base_url)
    ):
        raise HTTPException(
            status_code=400,
            detail="100% privacy mode requires a local embedding model — "
            "the configured Ollama server is remote",
        )

    config_defaults = AppConfig()
    config = AppConfig(
        db_path=Path(payload.db) if payload.db is not None else config_defaults.db_path,
        model_name=payload.model or config_defaults.model_name,
        chunk_chars=payload.chunk_chars or config_defaults.chunk_chars,
        overlap=payload.overlap or config_defaults.overlap,
    )
    resolved_db = config.resolve_db_path(Path.cwd())
    _ensure_db_parent(resolved_db)

    resolved_paths = _validate_paths(payload.paths)

    await asyncio.to_thread(
        _save_source_manifest,
        resolved_db,
        resolved_paths,
        payload.privacy,
        payload.exclude_paths,
    )

    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id": job_id,
        "status": "running",
        "processed": 0,
        "total": 0,
        "current_file": "",
        "stats": None,
        "error": None,
    }
    _index_jobs[job_id] = job

    exclude: frozenset[str] | None = (
        frozenset(payload.exclude_paths) if payload.exclude_paths else None
    )

    async def _run() -> None:
        try:
            result = await asyncio.to_thread(
                _run_index_job, resolved_paths, config, resolved_db, job, exclude, payload.privacy
            )
            job["status"] = "complete"
            job["stats"] = result
            _notify_indexing_done(result)
        except Exception as exc:
            LOGGER.exception("Indexing job %s failed: %s", job_id, exc)
            job["status"] = "error"
            job["error"] = str(exc)
            _notify_indexing_done(None, error=str(exc))

    asyncio.create_task(_run())
    return {"status": "ok", "job_id": job_id}


@app.get("/index/status/{job_id}")
async def get_index_status(job_id: str) -> dict[str, Any]:
    """Poll the status of a running or completed indexing job."""
    job = _index_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/system/info")
async def get_system_info() -> dict[str, Any]:
    """Return RAM + runtime backend and indexing strategy details."""
    memory = await asyncio.to_thread(_get_memory_info)
    runtime = await asyncio.to_thread(_get_runtime_info)
    return {**memory, **runtime}


class ScanPayload(BaseModel):
    paths: List[str]


@app.post("/index/scan")
async def scan_index_paths(payload: ScanPayload) -> dict[str, Any]:
    """Scan paths for PDFs and return file stats without indexing."""
    if not payload.paths:
        raise HTTPException(status_code=400, detail="No path provided")

    resolved_paths = _validate_paths(payload.paths)

    _LARGE_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

    def _scan() -> dict[str, Any]:
        from docfinder.utils.files import iter_document_paths

        docs = list(iter_document_paths(resolved_paths))
        total_size = 0
        large_files: list[dict[str, Any]] = []
        by_type: dict[str, int] = {}
        for f in docs:
            ext = f.suffix.lower()
            by_type[ext] = by_type.get(ext, 0) + 1
            try:
                size = f.stat().st_size
                total_size += size
                if size >= _LARGE_FILE_BYTES:
                    large_files.append({"name": f.name, "path": str(f), "size_bytes": size})
            except OSError:
                pass
        return {
            "file_count": len(docs),
            "total_size_bytes": total_size,
            "large_files": large_files,
            "by_type": by_type,
        }

    return await asyncio.to_thread(_scan)
