"""FastAPI application backing the DocFinder web UI."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
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

app = FastAPI(title="DocFinder Web", version="0.1.0")
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


@app.on_event("startup")
async def startup_event() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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

    embedder = EmbeddingModel(EmbeddingConfig(model_name=AppConfig().model_name))
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
        return {"documents": [], "stats": {"document_count": 0, "chunk_count": 0, "total_size_bytes": 0}}

    embedder = EmbeddingModel(EmbeddingConfig(model_name=AppConfig().model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        documents = store.list_documents()
        stats = store.get_stats()
    finally:
        store.close()
    
    return {"documents": documents, "stats": stats}


@app.delete("/documents/{doc_id}")
async def delete_document_by_id(doc_id: int, db: Path | None = None) -> dict[str, Any]:
    """Delete a document by its ID."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = EmbeddingModel(EmbeddingConfig(model_name=AppConfig().model_name))
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

    embedder = EmbeddingModel(EmbeddingConfig(model_name=AppConfig().model_name))
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


@app.delete("/documents/cleanup")
async def cleanup_missing_files(db: Path | None = None) -> dict[str, Any]:
    """Remove documents whose files no longer exist on disk."""
    resolved_db = _resolve_db_path(db)
    if not resolved_db.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    embedder = EmbeddingModel(EmbeddingConfig(model_name=AppConfig().model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    try:
        removed_count = store.remove_missing_files()
    finally:
        store.close()
    
    return {"status": "ok", "removed_count": removed_count}


def _run_index_job(paths: List[Path], config: AppConfig, resolved_db: Path) -> dict[str, Any]:
    embedder = EmbeddingModel(EmbeddingConfig(model_name=config.model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    indexer = Indexer(
        embedder,
        store,
        chunk_chars=config.chunk_chars,
        overlap=config.overlap,
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


@app.post("/index")
async def index_documents(payload: IndexPayload) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    sanitized_paths = [p.replace("\r", "").replace("\n", "") for p in payload.paths]
    logger.info("DEBUG: Received paths = %s", sanitized_paths)
    logger.info("DEBUG: Path type = %s", type(payload.paths))

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

    # Security: Define safe base directory for path traversal protection
    # User can only access directories within their home directory or an explicitly allowed path
    # For now, we allow access to the entire filesystem as the user is expected to be trusted
    # In production, you might want to restrict this to specific directories
    # IMPORTANT: Use canonical (real) path to prevent symlink-based bypasses
    safe_base_dir = Path(os.path.realpath(str(Path.home())))

    # Validate and resolve paths safely
    resolved_paths = []
    for p in payload.paths:
        # Sanitize input: remove newlines and carriage returns
        clean_path = p.strip().replace("\r", "").replace("\n", "")
        if not clean_path:
            continue

        # Security: Reject paths with null bytes or other dangerous characters
        if "\0" in clean_path:
            raise HTTPException(status_code=400, detail="Invalid path: contains null byte")

        try:
            # Step 1: Expand user directory first
            expanded_path = os.path.expanduser(clean_path)

            # Step 2: Use os.path.realpath for secure path resolution (prevents symlink attacks)
            # This also resolves relative paths and removes .. components
            real_path = os.path.realpath(expanded_path)

            # Step 3: Additional security check - verify it's an absolute path
            if not os.path.isabs(real_path):
                raise HTTPException(status_code=400, detail="Invalid path: must be absolute")

            # Step 4: CRITICAL SECURITY CHECK - Verify path is within safe base directory
            # We use canonical string prefix comparison for maximum robustness:
            # - Both paths are already fully resolved via os.path.realpath
            # - String prefix check works across all Python versions
            # - Avoids edge cases with is_relative_to() and symlinked parents
            # - Ensures path cannot escape the allowed directory (e.g., /etc/passwd)
            # Add path separator to prevent partial matches (e.g., /home/user vs /home/user2)
            safe_base_str = str(safe_base_dir) + os.sep
            real_path_str = real_path + os.sep

            if not real_path_str.startswith(safe_base_str):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: path is outside allowed directory",
                )

            # Step 5: Create Path object from the validated canonical path
            # This breaks the taint chain for CodeQL static analysis
            validated_path = Path(real_path)

            # Step 6: Now that path is validated, perform filesystem operations
            if not validated_path.exists():
                raise HTTPException(status_code=404, detail="Path not found: %s" % clean_path)

            # Step 7: Verify it's a directory (not a file)
            if not validated_path.is_dir():
                raise HTTPException(
                    status_code=400, detail="Path must be a directory: %s" % clean_path
                )

            resolved_paths.append(validated_path)

        except (ValueError, OSError) as e:
            logger.error("Invalid path '%s': %s", clean_path, e)
            raise HTTPException(status_code=400, detail="Invalid path: %s" % clean_path)

    try:
        stats = await asyncio.to_thread(_run_index_job, resolved_paths, config, resolved_db)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Indexing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "db": str(resolved_db), "stats": stats}
