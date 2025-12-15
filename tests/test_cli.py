"""Tests for CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from docfinder.cli import app, _setup_logging, _ensure_db_parent


runner = CliRunner()


class TestSetupLogging:
    """Tests for _setup_logging helper."""

    def test_setup_logging_verbose(self) -> None:
        """Verbose mode sets DEBUG level."""
        with patch("docfinder.cli.logging.basicConfig") as mock_config:
            _setup_logging(verbose=True)
            mock_config.assert_called_once()
            assert mock_config.call_args[1]["level"] == logging.DEBUG

    def test_setup_logging_normal(self) -> None:
        """Normal mode sets INFO level."""
        with patch("docfinder.cli.logging.basicConfig") as mock_config:
            _setup_logging(verbose=False)
            mock_config.assert_called_once()
            assert mock_config.call_args[1]["level"] == logging.INFO


class TestEnsureDbParent:
    """Tests for _ensure_db_parent helper."""

    def test_ensure_db_parent_creates_directory(self, tmp_path: Path) -> None:
        """Creates parent directory if it doesn't exist."""
        db_path = tmp_path / "subdir" / "test.db"
        assert not db_path.parent.exists()
        _ensure_db_parent(db_path)
        assert db_path.parent.exists()

    def test_ensure_db_parent_existing_directory(self, tmp_path: Path) -> None:
        """Does not fail if directory already exists."""
        db_path = tmp_path / "test.db"
        _ensure_db_parent(db_path)
        assert db_path.parent.exists()


class TestIndexCommand:
    """Tests for the index command."""

    def test_index_no_pdfs_found(self, tmp_path: Path) -> None:
        """Shows warning when no PDFs are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        db_path = tmp_path / "test.db"

        result = runner.invoke(app, ["index", str(empty_dir), "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No PDFs found" in result.stdout

    @patch("docfinder.cli.EmbeddingModel")
    @patch("docfinder.cli.SQLiteVectorStore")
    @patch("docfinder.cli.Indexer")
    def test_index_with_pdfs(
        self,
        mock_indexer_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successfully indexes PDFs."""
        # Create a fake PDF file
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        fake_pdf = pdf_dir / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake pdf content")

        db_path = tmp_path / "test.db"

        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_indexer = MagicMock()
        mock_stats = MagicMock()
        mock_stats.inserted = 1
        mock_stats.updated = 0
        mock_stats.skipped = 0
        mock_stats.failed = 0
        mock_indexer.index.return_value = mock_stats
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(app, ["index", str(pdf_dir), "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Inserted: 1" in result.stdout

    @patch("docfinder.cli.EmbeddingModel")
    @patch("docfinder.cli.SQLiteVectorStore")
    @patch("docfinder.cli.Indexer")
    def test_index_verbose(
        self,
        mock_indexer_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verbose flag is accepted."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        fake_pdf = pdf_dir / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake pdf")

        db_path = tmp_path / "test.db"

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_indexer = MagicMock()
        mock_stats = MagicMock()
        mock_stats.inserted = 1
        mock_stats.updated = 0
        mock_stats.skipped = 0
        mock_stats.failed = 0
        mock_indexer.index.return_value = mock_stats
        mock_indexer_class.return_value = mock_indexer

        result = runner.invoke(app, ["index", str(pdf_dir), "--db", str(db_path), "-v"])
        assert result.exit_code == 0


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_database_not_found(self, tmp_path: Path) -> None:
        """Raises error when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["search", "test query", "--db", str(db_path)])
        assert result.exit_code != 0
        # Error message is part of the output or in the exception repr
        output = result.stdout + result.stderr if result.stderr else result.stdout
        if result.exception:
            output += repr(result.exception)
        assert "Database not found" in output or result.exit_code == 2

    @patch("docfinder.cli.EmbeddingModel")
    @patch("docfinder.cli.SQLiteVectorStore")
    @patch("docfinder.cli.Searcher")
    def test_search_no_results(
        self,
        mock_searcher_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Shows message when no results found."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No matches found" in result.stdout

    @patch("docfinder.cli.EmbeddingModel")
    @patch("docfinder.cli.SQLiteVectorStore")
    @patch("docfinder.cli.Searcher")
    def test_search_with_results(
        self,
        mock_searcher_class: MagicMock,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Displays results in a table."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.path = "/path/to/doc.pdf"
        mock_result.chunk_index = 0
        mock_result.text = "This is a test snippet"

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [mock_result]
        mock_searcher_class.return_value = mock_searcher

        result = runner.invoke(app, ["search", "test query", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "0.9500" in result.stdout


class TestPruneCommand:
    """Tests for the prune command."""

    def test_prune_database_not_found(self, tmp_path: Path) -> None:
        """Shows message when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        result = runner.invoke(app, ["prune", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Database not found" in result.stdout

    @patch("docfinder.cli.EmbeddingModel")
    @patch("docfinder.cli.SQLiteVectorStore")
    def test_prune_success(
        self,
        mock_store_class: MagicMock,
        mock_embedder_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successfully prunes orphaned documents."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        mock_embedder = MagicMock()
        mock_embedder.dimension = 768
        mock_embedder_class.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.remove_missing_files.return_value = 3
        mock_store_class.return_value = mock_store

        result = runner.invoke(app, ["prune", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Removed 3 orphaned documents" in result.stdout


class TestWebCommand:
    """Tests for the web command."""

    def test_web_starts_server(
        self,
        tmp_path: Path,
    ) -> None:
        """Starts uvicorn server with correct parameters."""
        db_path = tmp_path / "test.db"
        db_path.touch()

        with patch("uvicorn.run") as mock_uvicorn_run:
            result = runner.invoke(
                app, ["web", "--host", "0.0.0.0", "--port", "9000", "--db", str(db_path)]
            )
            assert result.exit_code == 0
            mock_uvicorn_run.assert_called_once()
            call_kwargs = mock_uvicorn_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 9000

    def test_web_warns_missing_database(
        self,
        tmp_path: Path,
    ) -> None:
        """Shows warning when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["web", "--db", str(db_path)])
            assert result.exit_code == 0
            assert "database not found" in result.stdout.lower()
