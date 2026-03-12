"""RAG engine — coordinates semantic search with local LLM generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from docfinder.index.search import SearchResult, Searcher
from docfinder.index.storage import SQLiteVectorStore
from docfinder.rag.llm import LocalLLM, ModelSpec, ensure_model, select_model

logger = logging.getLogger(__name__)

# Rough token budget.  1 token ≈ 4 chars for English text.
_MAX_CONTEXT_TOKENS = 4000
_CHARS_PER_TOKEN = 4
_MAX_CONTEXT_CHARS = _MAX_CONTEXT_TOKENS * _CHARS_PER_TOKEN  # 16 000 chars

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided document context. "
    "Use ONLY the information from the context below to answer. "
    "If the context does not contain enough information, say so clearly. "
    "Answer in the same language as the user's question."
)


@dataclass(slots=True)
class RAGResult:
    """Result produced by the RAG pipeline."""

    answer: str
    sources: List[SearchResult]
    context_chunks: List[dict]


class RAGEngine:
    """Orchestrates retrieval (Searcher) and generation (LocalLLM).

    Usage::

        engine = RAGEngine.from_defaults(db_path, embedder)
        result = engine.query("What is the main finding?")
    """

    def __init__(
        self,
        searcher: Searcher,
        store: SQLiteVectorStore,
        llm: LocalLLM,
        *,
        window_size: int = 10,
    ) -> None:
        self.searcher = searcher
        self.store = store
        self.llm = llm
        self.window_size = window_size

    @classmethod
    def from_defaults(
        cls,
        db_path: Path,
        embedder: "docfinder.embedding.encoder.EmbeddingModel",  # noqa: F821
        *,
        model_spec: ModelSpec | None = None,
        models_dir: Path | None = None,
        window_size: int = 10,
        n_ctx: int = 4096,
    ) -> "RAGEngine":
        """Convenience constructor that wires everything together.

        * Auto-selects the best GGUF model for the current machine.
        * Downloads the model if not cached.
        * Opens the SQLite store and creates a Searcher.
        """
        store = SQLiteVectorStore(db_path, dimension=embedder.dimension)
        searcher = Searcher(embedder, store)

        if model_spec is None:
            model_spec = select_model()
        model_path = ensure_model(model_spec, models_dir)
        llm = LocalLLM(model_path, n_ctx=n_ctx)

        return cls(searcher, store, llm, window_size=window_size)

    # ── public API ──────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        top_k: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> RAGResult:
        """Run the full RAG pipeline: search → build context → generate answer."""
        results = self.searcher.search(question, top_k=top_k)
        if not results:
            return RAGResult(
                answer="No relevant documents found for your query.",
                sources=[],
                context_chunks=[],
            )

        context_chunks = self._build_context(results)
        context_text = self._assemble_context_text(context_chunks, results)
        answer = self._generate(question, context_text, max_tokens=max_tokens, temperature=temperature)

        return RAGResult(answer=answer, sources=results, context_chunks=context_chunks)

    # ── internals ───────────────────────────────────────────────────────

    def _build_context(self, results: List[SearchResult]) -> List[dict]:
        """Retrieve surrounding chunks for each search hit."""
        all_chunks: List[dict] = []
        seen: set[tuple[int, int]] = set()  # (document_id, chunk_index)

        for result in results:
            # We need the document_id — look it up via the store
            row = self.store.connection.execute(
                "SELECT id FROM documents WHERE path = ?", (str(result.path),)
            ).fetchone()
            if row is None:
                continue
            doc_id = row["id"]

            window = self.store.get_context_window(
                doc_id, result.chunk_index, self.window_size
            )
            for chunk in window:
                key = (doc_id, chunk["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    all_chunks.append({**chunk, "document_id": doc_id, "path": str(result.path)})

        return all_chunks

    @staticmethod
    def _assemble_context_text(chunks: List[dict], results: List[SearchResult]) -> str:
        """Join chunk texts, trimming symmetrically if the total exceeds the budget."""
        # Group by document path for readability
        by_doc: dict[str, List[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c["path"], []).append(c)

        parts: List[str] = []
        total_chars = 0
        for path, doc_chunks in by_doc.items():
            doc_chunks.sort(key=lambda c: c["chunk_index"])
            header = f"--- Document: {path} ---"
            parts.append(header)
            total_chars += len(header)

            for c in doc_chunks:
                text = c["text"]
                if total_chars + len(text) > _MAX_CONTEXT_CHARS:
                    remaining = _MAX_CONTEXT_CHARS - total_chars
                    if remaining > 100:
                        parts.append(text[:remaining] + " [...]")
                    break
                parts.append(text)
                total_chars += len(text)

            if total_chars >= _MAX_CONTEXT_CHARS:
                break

        return "\n\n".join(parts)

    def _generate(
        self,
        question: str,
        context: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based only on the context above:"
                ),
            },
        ]
        return self.llm.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def close(self) -> None:
        """Release the underlying SQLite connection."""
        self.store.close()
