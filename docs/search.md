# Semantic Search

DocFinder uses **semantic search** — it finds documents by meaning, not just keyword matching.

## How It Works

### 1. Text Extraction & Chunking

Documents are split into overlapping chunks (default 1200 characters with 200-character overlap). Each chunk preserves its page number and document origin for citation.

### 2. Embedding

Each chunk is converted into a dense vector embedding using a SentenceTransformer model. The embedding captures the semantic meaning of the text.

```python
# Internally, DocFinder uses:
model = SentenceTransformer("BAAI/bge-m3")
embedding = model.encode(chunk_text)
```

### 3. Storage

Embeddings and their text are stored in a local SQLite database with WAL mode enabled. Cosine similarity is computed in pure NumPy — no vector database extensions required.

### 4. Retrieval

When you search, your query is embedded with the same model and compared against all stored embeddings using cosine similarity:

```
score = cosine_similarity(query_embedding, doc_embedding)
```

The top-K results (default 10) are returned, ranked by similarity score.

### 5. Adaptive Indexing

DocFinder automatically selects an indexing strategy based on your machine's resources:

- **Multi-core systems**: Parallel processing via multiprocessing pool (Linux and Windows; macOS currently uses a single worker)
- **Lower-resource systems**: Single-process streaming
- **Batch size**: Auto-tuned based on available memory

## Paging Model

Different document formats have different natural page units:

| Format | Page Unit |
|--------|-----------|
| PDF | Real page numbers |
| PPTX | Slide numbers |
| EPUB | Chapter numbers |
| Markdown | Heading sections |
| DOCX, ODF | Every 10 paragraphs |
| TXT, HTML, DOC, PPT | Every ~3000 characters |

## Reranking

Search results are refined by a cross-encoder reranker: embedding-based retrieval narrows the candidates (3× top-K), then a more accurate model re-scores them. The reranker is enabled by default in the web UI.

## Configuration

Key parameters you can adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 1200 | Maximum characters per chunk |
| `overlap` | 200 | Character overlap between chunks |
| `top_k` | 10 | Number of search results to return |
