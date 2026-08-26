# AGENTS.md

Guidance for coding agents working with this repository.

## Commands

```bash
make setup          # Create .venv and install extras [dev,web,gui]
make test           # pytest -v with coverage (HTML/XML via pyproject addopts)
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
make format-check   # ruff format --check src/ tests/
make check-all      # lint + format-check + test
make run            # Launch desktop GUI (pywebview)
make run-web        # Launch web UI at http://127.0.0.1:8000
make serve-docs     # mkdocs serve (installs [docs] extra first)
make build-docs     # mkdocs build (CI runs --strict)
make build-{macos,windows,linux}   # Package via scripts/
make clean          # Remove build artifacts
```

Single test:
```bash
pytest tests/test_web_app.py -v
pytest tests/test_indexer.py::TestIndexer::test_method -v
```

Targets hardcode `.venv/bin/...`; activate `.venv` for bare pytest.

On Linux CI, PyTorch is installed CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Architecture

**Layout:** src layout; package `docfinder` in `src/docfinder/`. Version 2.1.3, license AGPL-3.0-or-later.

**Entry points:** `docfinder` CLI (cli.py / typer; commands `index`, `search`, `prune`, `web`) and `docfinder-gui` (gui.py: uvicorn in a thread + pywebview; macOS native SpotlightPanel via pyobjc, CGEventTap global hotkey). gui.py's `freeze_support()` + child-exit guard must stay before all other imports.

**Core pipeline:**
1. `ingestion/pdf_loader.py` — multi-format: PDF (PyMuPDF, tables→Markdown), TXT, MD, DOCX/DOC, ODT/ODP/ODG, PPTX/PPT, HTML, EPUB; `build_chunks(path, max_chars=1200, overlap=200)`; paged chunking per format
2. `embedding/encoder.py` — `EmbeddingModel` wraps SentenceTransformer, default `BAAI/bge-m3` (1024-dim); device auto-detect CUDA → MPS → ROCm → CPU; optional ONNX backend (OpenVINO only in type hints, no implementation)
3. `index/indexer.py` — `Indexer(embedder, store, *, chunk_chars, overlap, embed_batch_size, progress_callback: (processed, total, current_file))`; parallel parsing via ProcessPoolExecutor when ≥4 docs (macOS always 1 worker); adaptive embed batch by RAM (4–64); IndexStats
4. `index/storage.py` — `SQLiteVectorStore`; WAL; `meta` table stores embedding model; `ensure_embedding_model()` wipes the index when model changed; numpy cosine (`embeddings @ query`); dimension mismatch raises ValueError
5. `index/search.py` — `Searcher.search(query, *, top_k=10, folders=None)` → SearchResult; `index/reranker.py` `Reranker` (cross-encoder ms-marco-MiniLM-L-2-v2) always used by web `/search`

**RAG (`rag/`):** `engine.py` `RAGEngine` (context window / page-based context, n_ctx from model tier); `llm.py` `LocalLLM` (llama-cpp-python, gated behind `rag` extra — NOT in dev/web/gui; endpoints 503 when absent), `MODEL_TIERS` Qwen3.5-9B/4B/2B GGUF chosen by RAM, cache `~/.cache/docfinder/models`.

**Web layer (`web/app.py`):** FastAPI; lifespan preloads `EmbeddingModel` singleton (`_get_embedder()` double-checked lock; same for reranker); CORS wide open. Routes: `POST /search`, `/search/folders`, `/rag/models`, `/rag/download`, `/rag/download/status`, `/rag/chat`, `/open`, `/gui/spotlight/hide`, `/documents/delete`, `/settings`, `/index` (returns job_id, background via asyncio.create_task → to_thread), `/index/scan`; `GET /documents`, `/settings`, `/system/info`, `/index/status/{job_id}`, `/` + `/spotlight` (web/frontend.py serves templates).
- `/index` path validation: strips CR/LF, rejects NUL (400), realpath must be absolute (400), inside home (403), existing directory (404)
- Default DB: `~/Documents/DocFinder/docfinder.db` (frozen); dev uses `data/docfinder.db` only if it exists (it does in this checkout, gitignored)

**Frontend:** vanilla JS, no framework. `index.html` (`escHtml()` XSS escaping, indexing poll 600 ms, RAG poll 800 ms); `spotlight.html` (macOS overlay).

**Settings:** `settings.py` — JSON per OS (macOS `~/Library/Application Support/DocFinder/settings.json`), default hotkey `<alt>+d`, `hotkey_enabled`; `config.py` — `AppConfig(db_path, model_name, chunk_chars=1200, overlap=200)`.

## Key Constraints

- Python 3.10+ (no walrus in type hints; use `from __future__ import annotations`)
- `numpy>=1.26,<3` pinned for C-extension compatibility
- SQLite stdlib only — no sqlite-vec/FTS5; pure numpy cosine similarity
- Ruff: line-length 100, double quotes, target py310, select E,F,W,I; `tests/*` ignores E501
- pytest `--strict-markers`; coverage always collected
- License AGPL-3.0-or-later (changed from MIT at v1.1.1 for PyMuPDF)

## Gotchas

- Switching embedding model wipes chunks+documents on next index(); stale-index search → 409 "re-index your documents"
- `Indexer._should_parallelize()` is False whenever `embed_batch_size` is set — tests rely on this
- macOS "parallel" indexing = 1 worker (known ProcessPoolExecutor crash, TODO in code)
- `prune` CLI still loads/downloads the embedding model although unneeded
- `_preload_reranker()` is dead code — reranker loads lazily on first search
- CLI `--model` defaults evaluate `AppConfig()` at import time
- docs/changelog.md is a snippet include of CHANGELOG.md — don't edit it directly
