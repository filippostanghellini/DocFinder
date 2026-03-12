# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup          # Create .venv and install all extras [dev,web,gui]
make test           # Run pytest with coverage (term + HTML + XML)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make format-check   # Check formatting without modifying files
make check-all      # lint + format-check + test
make run            # Launch native desktop GUI (pywebview)
make run-web        # Launch web interface at http://127.0.0.1:8000
```

Single test:
```bash
pytest tests/test_web_app.py -v
pytest tests/test_indexer.py::TestIndexer::test_method -v
```

On Linux CI, PyTorch is installed CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Architecture

**Entry points:** `docfinder` (CLI via `cli.py` / typer) and `docfinder-gui` (`gui.py` spawns uvicorn in a thread, wraps FastAPI in pywebview).

**Core pipeline:**
1. `ingestion/pdf_loader.py` — PyMuPDF extracts text, splits into overlapping chunks (default 1200 chars, 200 overlap)
2. `embedding/encoder.py` — `EmbeddingModel` wraps SentenceTransformer; auto-detects CUDA → MPS → ROCm → CPU; optionally uses ONNX/CoreML backends
3. `index/indexer.py` — `Indexer` orchestrates PDF discovery, chunking, embedding, and storage; reports progress via callback `(processed, total, current_file)`
4. `index/storage.py` — `SQLiteVectorStore` persists chunks + embeddings; WAL mode; cosine similarity via numpy; batch inserts with `executemany()`
5. `index/search.py` — `Searcher` queries the store

**Web layer (`web/app.py`):**
- FastAPI app with lifespan-based startup preloading of `EmbeddingModel` singleton (thread-safe double-checked locking via `_get_embedder()`)
- `/index` returns a `job_id` immediately; background work runs via `asyncio.create_task`; poll `/index/status/{job_id}` for progress
- Default DB path: `~/Documents/DocFinder/docfinder.db` (frozen app) or `data/docfinder.db` (dev)
- Path validation uses `realpath` + home-dir prefix check

**Frontend (`web/templates/index.html`):** Vanilla JS single-page app; no framework. Polls indexing progress every 600 ms. Uses `escHtml()` for XSS prevention.

**Settings:** Hotkey config in `settings.py`; `AppConfig` in `config.py` handles paths and model name.

## Key Constraints

- Python 3.10+ required (no walrus operator in type hints; use `from __future__ import annotations`)
- `numpy<3` pinned for C-extension compatibility
- SQLite used with no extensions (no sqlite-vec, no FTS5 for search — pure numpy cosine similarity)
- Ruff line length: 100, double quotes, target py310
- Tests run with `--strict-markers`; coverage is always collected
