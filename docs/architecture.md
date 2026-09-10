# Architecture

## Overview

DocFinder is organized as a modular Python project with clear separation of concerns.

```
src/docfinder/
├── cli.py               # Typer CLI entry point
├── config.py             # AppConfig
├── gui.py                # Desktop GUI launcher (pywebview)
├── models.py             # Data models (ChunkRecord, etc.)
├── settings.py           # Hotkey & settings
├── embedding/
│   ├── encoder.py        # EmbeddingModel (SentenceTransformer)
├── index/
│   ├── indexer.py        # Indexer orchestration
│   ├── reranker.py       # Cross-encoder reranking
│   ├── search.py         # Searcher
│   └── storage.py        # SQLiteVectorStore
├── ingestion/
│   └── pdf_loader.py     # Text extraction & chunking
├── rag/
│   ├── engine.py         # RAG engine
│   └── llm.py            # LocalLLM wrapper
├── utils/
│   ├── files.py          # File discovery
│   ├── memory.py         # Memory detection
│   ├── notify.py         # OS notifications
│   └── text.py           # Text normalization & chunking
└── web/
    ├── app.py            # FastAPI application
    ├── frontend.py       # Static file serving
    └── templates/        # HTML templates
```

## Core Pipeline

```
File Discovery → Text Extraction → Chunking → Embedding → Storage → Search
```

### 1. File Discovery (`utils/files.py`)

Scans directories recursively, filtering by `SUPPORTED_EXTENSIONS`. Supports all common document formats.

### 2. Text Extraction (`ingestion/pdf_loader.py`)

Dispatches to the correct parser based on file extension. Each parser implements both a flat `iter_text_parts(path)` and a paged `iter_text_parts_*_paged(path)` variant.

### 3. Chunking (`utils/text.py`)

The `chunk_text_stream_paged` function takes a stream of `(page_num, text)` tuples and produces overlapping chunks with configurable size and overlap.

### 4. Embedding (`embedding/encoder.py`)

`EmbeddingModel` wraps SentenceTransformer with automatic backend detection:
- Detection order: CUDA → MPS → ROCm → CPU (MPS is preferred on Apple Silicon)
- Optional ONNX/OpenVINO backends via Optimum

### 5. Storage (`index/storage.py`)

`SQLiteVectorStore` persists chunks and embeddings using:
- WAL mode for concurrent access
- Cosine similarity via NumPy
- Batch inserts with `executemany()`
- A `meta` table recording the embedding model: indexes built with a different model are cleared automatically before re-indexing, and searching a stale index returns an explicit "re-index" error instead of failing cryptically

### 6. Search (`index/search.py`)

`Searcher` embeds the query, computes cosine similarity against all stored embeddings, and returns the top-K results.

## Desktop GUI Flow

```
pywebview window
    │
    ▼
Python backend (threaded)
    │
    ▼
FastAPI app (embedded uvicorn)
    │
    ▼
Core pipeline
```

The GUI launches a thread running uvicorn hosting the FastAPI app, then wraps it in a pywebview native window.

## Web Layer

The FastAPI app (`web/app.py`) provides:

**Indexing**
- `POST /index` — start indexing (returns `job_id`, async background processing)
- `GET /index/status/{job_id}` — poll indexing progress
- `POST /index/scan` — preview files before indexing

**Search**
- `POST /search` — semantic search (returns 409 with guidance if the index was built with a different embedding model)
- `GET /search/folders` — indexed directories for search-time filtering

**Documents**
- `GET /documents` — list indexed documents
- `DELETE /documents/{doc_id}` · `POST /documents/delete` · `DELETE /documents/cleanup` — remove documents
- `POST /open` — open a document with the system handler

**AI chat (RAG)**
- `POST /rag/chat` — answer a question from a chunk's context window
- `GET /rag/models` — available GGUF tiers
- `POST /rag/download` + `GET /rag/download/status` — model download

**Settings & system**
- `GET` / `POST /settings` · `GET /system/info`

(A few additional routes serve the desktop GUI internals.)

Model loading is thread-safe with double-checked locking:

```python
def _get_embedder():
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = EmbeddingModel()
    return _embedder
```
