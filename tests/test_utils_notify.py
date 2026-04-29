"""Tests for notification utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docfinder.utils.notify import send_notification


class TestSendNotification:
    """Tests for send_notification function."""

    def test_sends_macos_notification(self) -> None:
        """Calls osascript on macOS."""
        with patch("sys.platform", "darwin"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification("Test Title", "Test Message")

                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert args[0] == "osascript"
                assert "display notification" in args[2]
                assert "Test Title" in args[2]
                assert "Test Message" in args[2]

    def test_escapes_quotes_in_macos(self) -> None:
        """Escapes double quotes in macOS notification."""
        with patch("sys.platform", "darwin"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification('Title with "quotes"', 'Message with "quotes"')

                args = mock_popen.call_args[0][0]
                script = args[2]
                assert '\\"' in script  # Quotes should be escaped

    def test_sends_linux_notification(self) -> None:
        """Calls notify-send on Linux."""
        with patch("sys.platform", "linux"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification("Test Title", "Test Message")

                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert args[0] == "notify-send"
                assert args[1] == "Test Title"
                assert args[2] == "Test Message"

    def test_sends_windows_notification(self) -> None:
        """Calls PowerShell on Windows."""
        with patch("sys.platform", "win32"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification("Test Title", "Test Message")

                mock_popen.assert_called_once()
                args = mock_popen.call_args[0][0]
                assert args[0] == "powershell"
                assert "ToastNotification" in args[2]
                assert "Test Title" in args[2]

    def test_windows_uses_create_no_window_flag(self) -> None:
        """Uses CREATE_NO_WINDOW flag on Windows."""
        with patch("sys.platform", "win32"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification("Test", "Message")

                kwargs = mock_popen.call_args[1]
                assert kwargs["creationflags"] == 0x08000000

    def test_handles_unknown_platform(self) -> None:
        """Does nothing on unknown platforms."""
        with patch("sys.platform", "freebsd"):
            with patch("subprocess.Popen") as mock_popen:
                with patch("docfinder.utils.notify.LOGGER") as mock_logger:
                    send_notification("Test", "Message")

                    mock_popen.assert_not_called()
                    mock_logger.debug.assert_called_once()

    def test_handles_subprocess_error_silently(self) -> None:
        """Fails silently when subprocess fails."""
        with patch("sys.platform", "darwin"):
            with patch("subprocess.Popen", side_effect=Exception("Command failed")):
                with patch("docfinder.utils.notify.LOGGER") as mock_logger:
                    send_notification("Test", "Message")

                    mock_logger.debug.assert_called_once()

    def test_redirects_stdout_stderr(self) -> None:
        """Redirects stdout and stderr to DEVNULL."""
        with patch("sys.platform", "darwin"):
            with patch("subprocess.Popen") as mock_popen:
                send_notification("Test", "Message")

                kwargs = mock_popen.call_args[1]
                assert kwargs["stdout"] == -3  # subprocess.DEVNULL
                assert kwargs["stderr"] == -3
