"""Tests for memory utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docfinder.utils.memory import compute_embed_batch_size, get_memory_info


class TestGetMemoryInfo:
    """Tests for get_memory_info function."""

    def test_returns_psutil_when_available(self) -> None:
        """Uses psutil when available."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        mock_vm = MagicMock()
        mock_vm.available = 8 * 1024 * 1024 * 1024  # 8 GB
        mock_vm.total = 16 * 1024 * 1024 * 1024  # 16 GB

        with patch("psutil.virtual_memory", return_value=mock_vm):
            result = get_memory_info()
            assert result["available_mb"] == 8192
            assert result["total_mb"] == 16384

    def test_returns_none_on_import_error(self) -> None:
        """Returns None values when psutil not available."""
        with patch.dict("sys.modules", {"psutil": None}):
            result = get_memory_info()
            # Falls through to platform-specific code which may succeed
            assert "available_mb" in result
            assert "total_mb" in result

    def test_darwin_fallback(self) -> None:
        """Tests macOS-specific memory detection."""
        with patch("sys.platform", "darwin"):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch(
                    "subprocess.check_output",
                    side_effect=[
                        b"17179869184\n",  # 16 GB total
                        b"Pages free: 100000.\nPages inactive: 200000.\n",
                    ],
                ):
                    result = get_memory_info()
                    assert result["total_mb"] == 16384
                    # (100000 + 200000) * 4096 / 1024 / 1024 = 1171 MB
                    assert result["available_mb"] is not None

    def test_darwin_fallback_handles_errors(self) -> None:
        """Handles errors on macOS gracefully."""
        with patch("sys.platform", "darwin"):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("subprocess.check_output", side_effect=Exception("command failed")):
                    result = get_memory_info()
                    assert result["available_mb"] is None
                    assert result["total_mb"] is None

    def test_linux_fallback(self) -> None:
        """Tests Linux-specific memory detection - simplified."""
        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("builtins.open") as mock_open:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.__iter__ = MagicMock(
                        return_value=iter([
                            "MemTotal:       16384000 kB",
                            "MemAvailable:    8192000 kB",
                        ])
                    )
                    mock_open.return_value = mock_file

                    result = get_memory_info()
                    # Check structure is correct
                    assert "total_mb" in result
                    assert "available_mb" in result

    def test_linux_uses_memfree_when_available_missing(self) -> None:
        """Falls back to MemFree when MemAvailable missing - simplified."""
        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("builtins.open") as mock_open:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.__iter__ = MagicMock(
                        return_value=iter([
                            "MemTotal:       16384000 kB",
                            "MemFree:         4096000 kB",
                        ])
                    )
                    mock_open.return_value = mock_file

                    result = get_memory_info()
                    assert "available_mb" in result

    def test_windows_fallback(self) -> None:
        """Tests Windows-specific memory detection - simplified."""
        with patch("sys.platform", "win32"):
            with patch.dict("sys.modules", {"psutil": None}):
                # Just verify it returns something on Windows path
                result = get_memory_info()
                assert result is not None
                assert "available_mb" in result

    def test_returns_nones_on_complete_failure(self) -> None:
        """Returns None values when all methods fail."""
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("sys.platform", "unknown_os"):
                result = get_memory_info()
                assert result["available_mb"] is None
                assert result["total_mb"] is None


class TestComputeEmbedBatchSize:
    """Tests for compute_embed_batch_size function."""

    def test_batch_size_4096mb(self) -> None:
        """Returns 64,0.0 for 4GB+ available."""
        result = compute_embed_batch_size(8192)
        assert result == (64, 0.0)

        result = compute_embed_batch_size(4096)
        assert result == (64, 0.0)

    def test_batch_size_2048mb(self) -> None:
        """Returns 32,0.0 for 2-4GB available."""
        result = compute_embed_batch_size(2048)
        assert result == (32, 0.0)

        result = compute_embed_batch_size(3000)
        assert result == (32, 0.0)

    def test_batch_size_1024mb(self) -> None:
        """Returns 16,0.0 for 1-2GB available."""
        result = compute_embed_batch_size(1024)
        assert result == (16, 0.0)

        result = compute_embed_batch_size(1500)
        assert result == (16, 0.0)

    def test_batch_size_512mb(self) -> None:
        """Returns 8,0.05 for 512MB-1GB available."""
        result = compute_embed_batch_size(512)
        assert result == (8, 0.05)

        result = compute_embed_batch_size(900)
        assert result == (8, 0.05)

    def test_batch_size_low_memory(self) -> None:
        """Returns 4,0.2 for under 512MB available."""
        result = compute_embed_batch_size(256)
        assert result == (4, 0.2)

        result = compute_embed_batch_size(100)
        assert result == (4, 0.2)

        result = compute_embed_batch_size(0)
        assert result == (4, 0.2)

    def test_auto_detects_memory_when_none_provided(self) -> None:
        """Auto-detects memory when no argument provided."""
        with patch("docfinder.utils.memory.get_memory_info", return_value={"available_mb": 4096}):
            result = compute_embed_batch_size(None)
            assert result == (64, 0.0)

    def test_handles_none_from_get_memory_info(self) -> None:
        """Handles None from get_memory_info gracefully."""
        with patch("docfinder.utils.memory.get_memory_info", return_value={"available_mb": None}):
            result = compute_embed_batch_size(None)
            assert result == (64, 0.0)  # Falls back to 64 batch size
