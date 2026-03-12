"""FastAPI application backing the DocFinder web UI."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from docfinder.config import AppConfig
from docfinder.embedding.encoder import EmbeddingConfig, EmbeddingModel
from docfinder.index.indexer import Indexer
from docfinder.index.search import Searcher, SearchResult
from docfinder.index.storage import SQLiteVectorStore
from docfinder.settings import load_settings
from docfinder.settings import save_settings as _save_settings
from docfinder.web.frontend import router as frontend_router

LOGGER = logging.getLogger(__name__)

# ── Singleton EmbeddingModel ─────────────────────────────────────────────────
_embedder: EmbeddingModel | None = None
_embedder_lock = threading.Lock()


def _get_embedder() -> EmbeddingModel:
    """Return a cached EmbeddingModel, creating it on first call."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                config = AppConfig()
                _embedder = EmbeddingModel(EmbeddingConfig(model_name=config.model_name))
    return _embedder


# ── Async indexing job registry ───────────────────────────────────────────────
_index_jobs: dict[str, dict] = {}

# ── GUI callback registry (set by the desktop GUI layer, not used in web mode) ─
_spotlight_hide_callback: object = None  # callable | None


def register_spotlight_hide_callback(callback: object) -> None:
    """Register a callable that hides the spotlight panel (called by gui.py)."""
    global _spotlight_hide_callback
    _spotlight_hide_callback = callback


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Pre-load the embedding model at startup so the first request is instant
    await asyncio.to_thread(_get_embedder)
    yield


app = FastAPI(title="DocFinder Web", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(frontend_router)


class SearchPayload(BaseModel):
    query: str
    db: Path | None = None
    top_k: int = 10


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


def _resolve_db_path(db: Path | None) -> Path:
    config = AppConfig(db_path=db if db is not None else AppConfig().db_path)
    return config.resolve_db_path(Path.cwd())


def _ensure_db_parent(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


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
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    searcher = Searcher(embedder, store)
    results = searcher.search(query, top_k=top_k)
    store.close()
    return {"results": results}


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
    """Download (if needed) and load the RAG LLM.  Updates _rag_download state."""
    global _rag_llm
    from docfinder.rag.llm import _DEFAULT_MODELS_DIR, MODEL_TIERS, LocalLLM, ModelSpec

    # Pick the requested model or auto-select
    spec: ModelSpec | None = None
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
    _rag_llm = LocalLLM(local_path, n_ctx=4096)
    _rag_download["status"] = "ready"


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
        "Qwen2.5-7B-Instruct": "~4.7 GB",
        "Qwen2.5-3B-Instruct": "~2.1 GB",
        "Qwen2.5-1.5B-Instruct": "~1.1 GB",
    }
    return sizes.get(spec.name, "unknown")


@app.post("/rag/download")
async def rag_download(model_name: str | None = None) -> dict:
    """Start downloading and loading the RAG model in background."""
    if _rag_download["status"] in ("downloading", "loading"):
        return {"status": "already_running"}

    _rag_download["status"] = "downloading"
    _rag_download["error"] = None

    # Read user preference from settings
    settings = load_settings()
    chosen = model_name or settings.get("rag_model")

    async def _run():
        try:
            await asyncio.to_thread(_load_rag_llm, chosen)
        except Exception as exc:
            LOGGER.exception("RAG model download/load failed: %s", exc)
            _rag_download["status"] = "error"
            _rag_download["error"] = str(exc)

    asyncio.create_task(_run())
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
            "SELECT id FROM documents WHERE path = ?", (payload.document_path,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found in index")
        doc_id = row["id"]

        # Get context: try page-based first, fall back to fixed window
        import json as _json

        max_chars = 4000 * 4  # ~4000 tokens

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
        answer = await asyncio.to_thread(_rag_llm.chat, messages, max_tokens=1024, temperature=0.2)
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
            os.startfile(path)  # type: ignore[attr-defined]
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
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        removed_count = store.remove_missing_files()
    finally:
        store.close()

    return {"status": "ok", "removed_count": removed_count}


@app.delete("/documents/{doc_id}")
async def delete_document_by_id(doc_id: int, db: Path | None = None) -> dict[str, Any]:
    """Delete a document by its ID."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = _get_embedder()
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        deleted = store.delete_document(doc_id)
    finally:
        store.close()

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
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        if payload.doc_id is not None:
            deleted = store.delete_document(payload.doc_id)
        else:
            deleted = store.delete_document_by_path(payload.path)  # type: ignore
    finally:
        store.close()

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "ok"}


@app.get("/settings")
async def get_settings() -> dict:
    """Return current user settings."""
    return load_settings()


@app.post("/settings")
async def update_settings(payload: SettingsPayload) -> dict:
    """Persist updated settings and return the full settings dict."""
    current = load_settings()
    if payload.hotkey is not None:
        current["hotkey"] = payload.hotkey
    if payload.hotkey_enabled is not None:
        current["hotkey_enabled"] = payload.hotkey_enabled
    if payload.rag_enabled is not None:
        current["rag_enabled"] = payload.rag_enabled
    if payload.rag_model is not None:
        current["rag_model"] = payload.rag_model
    _save_settings(current)
    return current


def _compute_embed_batch_size() -> int:
    """Choose embedding batch size based on available RAM to avoid OOM."""
    info = _get_memory_info()
    available = info.get("available_mb")
    if available is None or available >= 4096:
        return 32
    elif available >= 2048:
        return 16
    else:
        return 8


def _run_index_job(
    paths: List[Path],
    config: AppConfig,
    resolved_db: Path,
    job: dict | None = None,
    exclude_paths: frozenset[str] | None = None,
) -> dict[str, Any]:
    embedder = _get_embedder()
    embed_batch_size = _compute_embed_batch_size()

    def _progress(processed: int, total: int, current_file: str) -> None:
        if job is not None:
            job["processed"] = processed
            job["total"] = total
            job["current_file"] = current_file

    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    indexer = Indexer(
        embedder,
        store,
        chunk_chars=config.chunk_chars,
        overlap=config.overlap,
        embed_batch_size=embed_batch_size,
        progress_callback=_progress,
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
    """Return available and total RAM in MB using platform-native methods. No extra deps."""
    try:
        import psutil  # optional – used if installed

        vm = psutil.virtual_memory()
        return {
            "available_mb": vm.available // (1024 * 1024),
            "total_mb": vm.total // (1024 * 1024),
        }
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            import subprocess as _sp

            total = int(_sp.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            vm_out = _sp.check_output(["vm_stat"]).decode()
            free_pages = inactive_pages = 0
            for line in vm_out.splitlines():
                if line.startswith("Pages free:"):
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))
            available = (free_pages + inactive_pages) * 4096
            return {"available_mb": available // (1024 * 1024), "total_mb": total // (1024 * 1024)}
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo") as _f:
                for line in _f:
                    k, v = line.split(":")
                    meminfo[k.strip()] = int(v.strip().split()[0])  # kB
            return {
                "available_mb": meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) // 1024,
                "total_mb": meminfo.get("MemTotal", 0) // 1024,
            }
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import ctypes

            class _MemStatEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatEx()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return {
                "available_mb": stat.ullAvailPhys // (1024 * 1024),
                "total_mb": stat.ullTotalPhys // (1024 * 1024),
            }
        except Exception:
            pass

    return {"available_mb": None, "total_mb": None}


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
                _run_index_job, resolved_paths, config, resolved_db, job, exclude
            )
            job["status"] = "complete"
            job["stats"] = result
        except Exception as exc:
            LOGGER.exception("Indexing job %s failed: %s", job_id, exc)
            job["status"] = "error"
            job["error"] = str(exc)

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
    """Return available and total RAM in MB for the host machine."""
    return await asyncio.to_thread(_get_memory_info)


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
