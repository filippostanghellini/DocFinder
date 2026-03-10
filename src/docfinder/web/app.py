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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Pre-load the embedding model at startup so the first request is instant
    await asyncio.to_thread(_get_embedder)
    yield


app = FastAPI(title="DocFinder Web", version="1.1.1", lifespan=lifespan)
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


def _run_index_job(
    paths: List[Path],
    config: AppConfig,
    resolved_db: Path,
    job: dict | None = None,
) -> dict[str, Any]:
    embedder = _get_embedder()

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
        progress_callback=_progress,
    )
    try:
        stats = indexer.index(paths)
    finally:
        store.close()

    return {
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "processed_files": [str(path) for path in stats.processed_files],
    }


def _validate_index_paths(payload: "IndexPayload") -> List[Path]:
    """Validate and resolve paths from an IndexPayload. Raises HTTPException on error."""
    logger = logging.getLogger(__name__)
    safe_base_dir = Path(os.path.realpath(str(Path.home())))
    resolved_paths: List[Path] = []

    for p in payload.paths:
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

    resolved_paths = _validate_index_paths(payload)

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
            result = await asyncio.to_thread(
                _run_index_job, resolved_paths, config, resolved_db, job
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
