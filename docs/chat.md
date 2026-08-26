# AI Chat

DocFinder includes a local AI chat feature that lets you ask questions about your indexed documents and get precise answers.

## How It Works

### Architecture

```
Selected Chunk
(document_path + chunk_index)
    │
    ▼
Context Window ──► the chunk's page (fallback: ±10 surrounding chunks)
    │
    ▼
LLM (local)
    │
    ▼
Grounded Answer
```

1. You reference a specific chunk of a document (`document_path` + `chunk_index` — the UI fills these for you)
2. DocFinder builds the context from that chunk's page, or from a window of ±10 surrounding chunks as fallback
3. The context plus your question are sent to a local LLM
4. The LLM generates an answer grounded in that context only

### Local LLM

DocFinder uses **Qwen3.5 models** via `llama-cpp-python` (GGUF format). The model is selected automatically from your available RAM; GPU offload (Metal on Apple Silicon, CUDA on NVIDIA) is enabled when possible:

- **≥16 GB RAM**: Qwen3.5-9B (~5.7 GB download)
- **8–16 GB RAM**: Qwen3.5-4B (~2.7 GB download)
- **any machine**: Qwen3.5-2B (~1.3 GB download)

Reasoning ("thinking") output is disabled by default in these models, so answers stay concise and grounded.

### RAG Engine

The RAG engine (`src/docfinder/rag/engine.py`) constructs prompts with strict grounding instructions:

> "Use ONLY the information from the context below to answer. If the context does not contain enough information, say so clearly."

This ensures the AI doesn't hallucinate — answers are always based on your actual documents.

## Requirements

To use the AI Chat feature, install the RAG extra:

```bash
pip install -e ".[rag]"
```

This installs `llama-cpp-python`, which provides local LLM inference.

> Note: `make setup` installs the `dev`, `web` and `gui` extras but **not** `rag` — install it explicitly to enable AI chat.

## Usage

### Desktop GUI

Open the chat panel to start asking questions. Type a question about your indexed documents and DocFinder will search and answer.

### Web Interface

The web UI at `http://127.0.0.1:8000` includes a chat interface with the same functionality.

### API

The FastAPI backend exposes a chat endpoint:

```bash
curl -X POST http://127.0.0.1:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the document say about budget allocation?",
    "document_path": "/path/to/document.pdf",
    "chunk_index": 3
  }'
```

The question is answered using the context window surrounding the given chunk (`chunk_index`) of the specified document.

## Model Auto-Selection

DocFinder checks your system once at startup:

| Available RAM | Model | Download |
|---------------|-------|----------|
| ≥16 GB | Qwen3.5-9B | ~5.7 GB |
| 8–16 GB | Qwen3.5-4B | ~2.7 GB |
| any machine | Qwen3.5-2B | ~1.3 GB |

GPU acceleration (Metal / CUDA) is used automatically when available; otherwise the model runs on CPU.
