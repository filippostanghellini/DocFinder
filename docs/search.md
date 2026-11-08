# Semantic & Hybrid Search

DocFinder combines **multimodal vector embeddings**, **exact-word & filename lexical boosting**, and **cross-encoder reranking** so you find files both by what they mean and by exact identifiers.

## How It Works

### 1. Structure-Aware Extraction & Chunking

Documents are split into overlapping chunks (default 1200 characters with 200-character overlap) while preserving structural provenance:

- **PDFs & Office:** Page, slide, or worksheet numbers
- **Source Code:** Exact `line_start` and `line_end` boundaries
- **Audio & Video:** 30-second windows with `start_sec` and `timestamp` (`M:SS`)

### 2. Modular Multimodal Embedding (EmbeddingGemma 2)

Each chunk is encoded into a 768-dimensional vector using `google/embeddinggemma-2`:

- **Asymmetric prefixes:** Document chunks are formatted with `title: {title} | text: {chunk}`, while queries use task-specific instructions (`task: search result | query: …` or `task: code retrieval | query: …` when prefixed with `code:`).
- **Safe precision:** Automatically runs in `bfloat16` (Apple Silicon MPS / Ampere+ CUDA) or `float32`, avoiding `float16` overflow.
- **Shared vector space:** Text (`270M`), Vision (`+170M`), and Audio (`+300M`) encoders map into the exact same 768-d space. Enabling Vision or Audio later never invalidates your existing index.

```python
# Internally, DocFinder uses:
model = SentenceTransformer("google/embeddinggemma-2")
```

!!! note "Custom embedding models via Ollama"

    In **Settings → Models & Ollama** you can connect to a local or remote Ollama server.
    When using EmbeddingGemma 2 on Ollama, DocFinder automatically picks the tag matching your
    enabled modalities (`embeddinggemma-2:270m`, `440m`, `570m`, or `740m`).

### 3. Hybrid Retrieval: Cosine Similarity + Exact-Word Boost

Pure vector search captures meaning across 100+ languages, but no embedding holds an exact invoice number (`INV-2026-089`) or function name (`refreshAccessToken`) as reliably as string matching.

At query time, `SQLiteVectorStore.search()` computes:

```
score = cosine_similarity(query_vec, chunk_vec) + lexical_boost(query_text, chunk, title, filename)
```

Exact substring matches and token hits in the chunk text, document title, or filename receive an automatic boost before top-K selection.

### 4. Cross-Encoder Reranking

When searching from the Web UI or Spotlight, the bi-encoder fetches `max(top_k * 3, 30)` candidates and passes them to `cross-encoder/ms-marco-MiniLM-L-2-v2` for fine-grained re-scoring (automatically skipped when all candidates have very low similarity to save latency).

## Paging & Location Model

| Format | Location Badge | Open Behavior |
|--------|----------------|---------------|
| PDF | `Page N` | Opens the file with the system default handler. The `Page N` badge is informational (the app does not jump to a specific page on open) |
| Source Code | `Lines A-B` | Opens directly at `file:line` in VS Code, Cursor, or Zed when one is installed; otherwise opens with the system default handler |
| Audio / Video | `M:SS` | Opens the file with the system default handler. The `M:SS` badge marks the matched 30-second window |
| PPTX / XLSX / EPUB | `Page N` | Slide, worksheet, or chapter number shown as a badge; file opens with the system default handler |
| Markdown / DOCX / ODF | `Page N` | Heading section or 10-paragraph group shown as a badge; file opens with the system default handler |

Every result row also has a **Show in Folder** action (and `Cmd+Enter` / `Ctrl+Enter` in Spotlight) that reveals and selects the file in Finder, Explorer, or your file manager.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 1200 | Maximum characters per text/code chunk |
| `overlap` | 200 | Character overlap between consecutive chunks |
| `top_k` | 10 | Number of search results to return |
