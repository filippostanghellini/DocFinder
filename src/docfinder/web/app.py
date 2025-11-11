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
        raise HTTPException(status_code=404, detail=f"Database not found: {resolved_db}")

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
    logger.info(f"DEBUG: Received paths = {payload.paths}")

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

    # Validate and resolve paths safely
    resolved_paths = []
    for p in payload.paths:
        # Sanitize input: remove newlines and carriage returns
        clean_path = p.strip().replace('\r', '').replace('\n', '')
        if not clean_path:
            continue
        
        try:
            # Expand ~ and resolve to absolute path
            resolved = Path(clean_path).expanduser().resolve()
            
            # Verify path exists
            if not resolved.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Path not found: '{clean_path}'"
                )
            
            # Verify it's a directory (not a file)
            if not resolved.is_dir():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Path must be a directory: '{clean_path}'"
                )
            
            resolved_paths.append(resolved)
            
        except (ValueError, OSError) as e:
            logger.error(f"Invalid path '{clean_path}': {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid path: '{clean_path}'"
            )

    try:
        stats = await asyncio.to_thread(_run_index_job, resolved_paths, config, resolved_db)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Indexing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "db": str(resolved_db), "stats": stats}
