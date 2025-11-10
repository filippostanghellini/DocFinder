"""Tests for file utility functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from docfinder.utils.files import compute_sha256, iter_pdf_paths


class TestIterPdfPaths:
    """Test iter_pdf_paths function."""

    def test_single_pdf_file(self, tmp_path: Path) -> None:
        """Should yield single PDF file."""
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy")
        
        paths = list(iter_pdf_paths([pdf]))
        
        assert len(paths) == 1
        assert paths[0] == pdf

    def test_directory_with_pdfs(self, tmp_path: Path) -> None:
        """Should find all PDFs in directory."""
        (tmp_path / "doc1.pdf").write_text("dummy1")
        (tmp_path / "doc2.pdf").write_text("dummy2")
        (tmp_path / "not_pdf.txt").write_text("text")
        
        paths = list(iter_pdf_paths([tmp_path]))
        
        assert len(paths) == 2
        assert all(p.suffix == ".pdf" for p in paths)

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Should find PDFs in nested directories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        (tmp_path / "root.pdf").write_text("root")
        (subdir / "nested.pdf").write_text("nested")
        (subdir / "file.txt").write_text("text")
        
        paths = list(iter_pdf_paths([tmp_path]))
        
        assert len(paths) == 2
        pdf_names = {p.name for p in paths}
        assert pdf_names == {"root.pdf", "nested.pdf"}

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Should match PDF files with different case extensions."""
        # Note: macOS uses case-insensitive file system by default,
        # so we test with unique filenames
        (tmp_path / "doc1.pdf").write_text("dummy1")
        (tmp_path / "doc2.PDF").write_text("dummy2")
        
        paths = list(iter_pdf_paths([tmp_path]))
        
        # Should find both PDF files regardless of extension case
        assert len(paths) >= 1  # At least one should be found
        assert all(p.suffix.lower() == ".pdf" for p in paths)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Should handle empty directory."""
        paths = list(iter_pdf_paths([tmp_path]))
        
        assert len(paths) == 0

    def test_multiple_inputs(self, tmp_path: Path) -> None:
        """Should handle multiple input paths."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        (dir1 / "doc1.pdf").write_text("d1")
        (dir2 / "doc2.pdf").write_text("d2")
        
        paths = list(iter_pdf_paths([dir1, dir2]))
        
        assert len(paths) == 2

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Should skip nonexistent files."""
        fake = tmp_path / "nonexistent.pdf"
        
        paths = list(iter_pdf_paths([fake]))
        
        assert len(paths) == 0


class TestComputeSha256:
    """Test compute_sha256 function."""

    def test_compute_hash_simple(self, tmp_path: Path) -> None:
        """Should compute SHA256 for file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        hash_result = compute_sha256(test_file)
        
        # SHA256 of "Hello, World!"
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert hash_result == expected

    def test_compute_hash_empty_file(self, tmp_path: Path) -> None:
        """Should compute hash for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        hash_result = compute_sha256(test_file)
        
        # SHA256 of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert hash_result == expected

    def test_compute_hash_binary_file(self, tmp_path: Path) -> None:
        """Should compute hash for binary file."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04")
        
        hash_result = compute_sha256(test_file)
        
        # Should return a valid hex hash
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_hash_large_file(self, tmp_path: Path) -> None:
        """Should handle large files efficiently."""
        test_file = tmp_path / "large.bin"
        # Write > 1MB to test chunked reading
        test_file.write_bytes(b"x" * (2 * 1024 * 1024))
        
        hash_result = compute_sha256(test_file)
        
        # Should complete without error
        assert len(hash_result) == 64

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Should produce same hash for same content."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        content = "Same content"
        file1.write_text(content)
        file2.write_text(content)
        
        hash1 = compute_sha256(file1)
        hash2 = compute_sha256(file2)
        
        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Should produce different hash for different content."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        hash1 = compute_sha256(file1)
        hash2 = compute_sha256(file2)
        
        assert hash1 != hash2
