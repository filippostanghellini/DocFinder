"""Command line interface for DocFinder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from docfinder.config import AppConfig
from docfinder.embedding.encoder import EmbeddingConfig, EmbeddingModel
from docfinder.index.indexer import Indexer
from docfinder.index.search import Searcher
from docfinder.index.storage import SQLiteVectorStore
from docfinder.utils.files import iter_pdf_paths
from docfinder.web.app import app as web_app


console = Console()
app = typer.Typer(help="DocFinder - local semantic search for PDFs")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _ensure_db_parent(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


@app.command()
def index(
    inputs: List[Path] = typer.Argument(
        ..., help="Paths with PDFs to index.", resolve_path=True
    ),
    db: Path = typer.Option(None, "--db", help="SQLite database path"),
    model: str = typer.Option(AppConfig().model_name, help="Sentence-transformer model name"),
    chunk_chars: int = typer.Option(AppConfig().chunk_chars, help="Chunk size in characters"),
    overlap: int = typer.Option(AppConfig().overlap, help="Chunk overlap"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Index one or more paths containing PDF files."""
    _setup_logging(verbose)
    config = AppConfig(
        db_path=db if db is not None else AppConfig().db_path,
        model_name=model,
        chunk_chars=chunk_chars,
        overlap=overlap,
    )

    resolved_db = config.resolve_db_path(Path.cwd())
    _ensure_db_parent(resolved_db)

    embedder = EmbeddingModel(EmbeddingConfig(model_name=config.model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    indexer = Indexer(embedder, store, chunk_chars=config.chunk_chars, overlap=config.overlap)

    console.print(f"Indexing into [bold]{resolved_db}[/bold]...")
    pdf_paths = list(iter_pdf_paths(inputs))
    if not pdf_paths:
        console.print("[yellow]No PDFs found.[/yellow]")
        store.close()
        return

    stats = indexer.index(pdf_paths)
    console.print(
        f"Inserted: {stats.inserted}, updated: {stats.updated}, "
        f"skipped: {stats.skipped}, failed: {stats.failed}"
    )
    store.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text"),
    db: Path = typer.Option(None, "--db", help="SQLite database path"),
    model: str = typer.Option(AppConfig().model_name, help="Sentence-transformer model name"),
    top_k: int = typer.Option(10, help="Number of results to display"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Execute a semantic search."""
    _setup_logging(verbose)
    config = AppConfig(db_path=db if db is not None else AppConfig().db_path, model_name=model)
    resolved_db = config.resolve_db_path(Path.cwd())

    if not resolved_db.exists():
        raise typer.BadParameter(f"Database not found: {resolved_db}")

    embedder = EmbeddingModel(EmbeddingConfig(model_name=config.model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    searcher = Searcher(embedder, store)

    results = searcher.search(query, top_k=top_k)
    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Score")
    table.add_column("Document")
    table.add_column("Chunk")
    table.add_column("Snippet")

    for result in results:
        snippet = result.text.replace("\n", " ")
        table.add_row(f"{result.score:.4f}", str(result.path), str(result.chunk_index), snippet[:180])

    console.print(table)
    store.close()


@app.command()
def prune(
    db: Path = typer.Option(None, "--db", help="SQLite database path"),
    model: Optional[str] = typer.Option(None, help="Optional, kept for interface parity"),
) -> None:
    """Remove documents that no longer exist on disk."""
    config = AppConfig(db_path=db if db is not None else AppConfig().db_path, model_name=model or AppConfig().model_name)
    resolved_db = config.resolve_db_path(Path.cwd())

    if not resolved_db.exists():
        console.print("[yellow]Database not found, nothing to prune.[/yellow]")
        return

    embedder = EmbeddingModel(EmbeddingConfig(model_name=config.model_name))
    store = SQLiteVectorStore(resolved_db, dimension=embedder.dimension)
    removed = store.remove_missing_files()
    console.print(f"Removed {removed} orphaned documents.")
    store.close()


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host interface"),
    port: int = typer.Option(8000, help="Server port"),
    db: Path = typer.Option(None, "--db", help="SQLite database path"),
) -> None:
    """Start the web interface."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(
            "uvicorn is not installed. Install the web extras with \"python -m pip install '.[web]'\""
        ) from exc

    config = AppConfig(db_path=db if db is not None else AppConfig().db_path)
    resolved_db = config.resolve_db_path(Path.cwd())
    if not resolved_db.exists():
        console.print("[yellow]Warning: database not found, searches might fail.[/yellow]")

    console.print(
        f"Starting web interface on http://{host}:{port} (database: {resolved_db})"
    )
    uvicorn.run(
        web_app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
