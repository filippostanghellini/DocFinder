"""SQLite + VSS vector store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from docfinder.models import ChunkRecord, DocumentMetadata


class SQLiteVectorStore:
    """Persistence layer for document and chunk embeddings."""

    def __init__(self, db_path: Path, *, dimension: int) -> None:
        self.db_path = Path(db_path)
        self.dimension = dimension
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _ensure_schema(self) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS documents_updated
                AFTER UPDATE ON documents
                BEGIN
                    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id)
                """
            )
            # Clean up legacy vector tables if present
            conn.execute("DROP TABLE IF EXISTS chunk_index")
            conn.execute("DROP TABLE IF EXISTS chunk_index_data")
            conn.execute("DROP TABLE IF EXISTS chunk_index_index")

            columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "embedding" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")

    def init_document(self, document: DocumentMetadata) -> tuple[int, str]:
        """Initialize a document for insertion.
        
        Returns:
            (doc_id, status) where status is 'inserted', 'updated', or 'skipped'.
            If skipped, doc_id is -1.
        """
        # Note: This should be called within a transaction
        conn = self._conn
        
        existing = conn.execute(
            "SELECT id, sha256 FROM documents WHERE path = ?",
            (str(document.path),),
        ).fetchone()

        if existing and existing["sha256"] == document.sha256:
            return -1, "skipped"

        if existing:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (existing["id"],))
            conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))

        doc_id = conn.execute(
            """
            INSERT INTO documents(path, title, sha256, mtime, size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(document.path),
                document.title,
                document.sha256,
                document.mtime,
                document.size,
            ),
        ).lastrowid
        
        return doc_id, "updated" if existing else "inserted"

    def insert_chunks(
        self, 
        doc_id: int, 
        chunks: Sequence[ChunkRecord], 
        embeddings: np.ndarray
    ) -> None:
        """Insert a batch of chunks for a document."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks length mismatch")
            
        conn = self._conn
        for chunk, vector in zip(chunks, embeddings):
            conn.execute(
                """
                INSERT INTO chunks(document_id, chunk_index, text, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    chunk.index,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=True),
                    sqlite3.Binary(np.asarray(vector, dtype="float32").tobytes()),
                ),
            )

    def upsert_document(
        self,
        document: DocumentMetadata,
        chunks: Sequence[ChunkRecord],
        embeddings: np.ndarray,
    ) -> str:
        with self.transaction():
            doc_id, status = self.init_document(document)
            if status == "skipped":
                return status
            
            self.insert_chunks(doc_id, chunks, embeddings)
            return status

    def search(self, embedding: np.ndarray, *, top_k: int = 10) -> List[dict]:
        query = np.asarray(embedding, dtype="float32")
        rows = self._conn.execute(
            """
            SELECT
                d.path AS path,
                d.title AS title,
                c.chunk_index AS chunk_index,
                c.text AS text,
                c.metadata AS metadata,
                c.embedding AS embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()

        if not rows:
            return []

        embeddings = np.vstack([np.frombuffer(row["embedding"], dtype="float32") for row in rows])
        scores = embeddings @ query

        if top_k < len(scores):
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        results: List[dict] = []
        for idx in top_indices:
            row = rows[idx]
            results.append(
                {
                    "path": row["path"],
                    "title": row["title"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "metadata": row["metadata"],
                    "score": float(scores[idx]),
                }
            )
        return results

    def remove_missing_files(self) -> int:
        """Remove documents whose files no longer exist."""
        with self.transaction() as conn:
            rows = conn.execute("SELECT id, path FROM documents").fetchall()
            missing = [row for row in rows if not Path(row["path"]).exists()]
            for row in missing:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
                conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
        return len(missing)
