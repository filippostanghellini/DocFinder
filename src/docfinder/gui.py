"""Desktop GUI launcher for DocFinder using pywebview.

This module provides a native desktop window that wraps the FastAPI web interface,
allowing users to interact with DocFinder through a graphical interface without
needing to use the command line.
"""

from __future__ import annotations

# CRITICAL: These must be at the very top, before any other imports
# to prevent multiprocessing child processes from spawning new windows
import multiprocessing
import sys

# Handle multiprocessing freeze support IMMEDIATELY
# This must happen before ANY other imports to prevent spawned processes
# from re-executing the entire application on macOS/Windows
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Detect if we're a multiprocessing child process and exit early
# This prevents child processes from creating GUI windows
if multiprocessing.current_process().name != "MainProcess":
    sys.exit(0)

import logging
import os
import socket
import threading
import time
from pathlib import Path

# Disable tokenizers parallelism to prevent forking issues with PyInstaller
# This must be set before any imports of transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Set multiprocessing start method to spawn on all platforms for PyInstaller compatibility
# This prevents issues with forked processes re-executing the main script
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set


def _get_log_file_path() -> Path:
    """Get path to log file for debugging startup issues."""
    if sys.platform == "win32":
        # On Windows, use LOCALAPPDATA for logs
        import os

        appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        log_dir = Path(appdata) / "DocFinder" / "logs"
    elif sys.platform == "darwin":
        log_dir = Path.home() / "Library" / "Logs" / "DocFinder"
    else:
        log_dir = Path.home() / ".local" / "share" / "docfinder" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "docfinder.log"


def _setup_logging() -> None:
    """Configure logging to both console and file."""
    log_file = _get_log_file_path()

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler - always logs DEBUG level for troubleshooting
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# Setup logging early
_setup_logging()
logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _get_icon_path() -> str | None:
    """Get the path to the application icon."""
    # When running from PyInstaller bundle
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # When running from source
        base_path = Path(__file__).parent.parent.parent.parent

    # Try different icon locations
    icon_candidates = [
        base_path / "Logo.png",
        base_path / "resources" / "Logo.png",
        Path(__file__).parent.parent.parent.parent / "Logo.png",
    ]

    for icon_path in icon_candidates:
        if icon_path.exists():
            return str(icon_path)

    return None


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for the server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    return False


class ServerThread(threading.Thread):
    """Thread that runs the uvicorn server."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = None

    def run(self) -> None:
        """Start the uvicorn server."""
        import uvicorn

        from docfinder.web.app import app

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self) -> None:
        """Signal the server to stop."""
        if self.server:
            self.server.should_exit = True


def main() -> None:
    """Launch the DocFinder desktop application."""
    logger.info("DocFinder starting up...")
    logger.info("Platform: %s, Python: %s", sys.platform, sys.version)

    try:
        import webview

        logger.debug("pywebview version: %s", getattr(webview, "__version__", "unknown"))
    except ImportError as exc:
        logger.error(
            "pywebview is not installed. Install the gui extras with: pip install 'docfinder[gui]'"
        )
        raise SystemExit(1) from exc

    try:
        # Find a free port
        host = "127.0.0.1"
        port = _find_free_port()
        url = f"http://{host}:{port}"

        logger.info("Starting DocFinder server on %s", url)

        # Start the server in a background thread
        server_thread = ServerThread(host, port)
        server_thread.start()

        # Wait for server to be ready
        if not _wait_for_server(host, port):
            logger.error("Server failed to start within timeout")
            raise SystemExit(1)

        logger.info("Server ready, launching window...")

        # Get icon path
        icon_path = _get_icon_path()

        # Create and start the webview window
        # Note: pywebview doesn't support setting window icon on all platforms
        # macOS uses the app bundle icon, Windows can use the icon parameter
        # Linux pywebview doesn't support icon parameter
        window_kwargs: dict = {
            "title": "DocFinder",
            "url": url,
            "width": 1200,
            "height": 800,
            "min_size": (800, 600),
            "resizable": True,
            "text_select": True,
        }

        # Add icon only on Windows (macOS uses app bundle, Linux doesn't support it)
        if icon_path and sys.platform == "win32":
            window_kwargs["icon"] = icon_path

        window = webview.create_window(**window_kwargs)

        def on_closed() -> None:
            """Handle window close event."""
            logger.info("Window closed, shutting down server...")
            server_thread.stop()

        window.events.closed += on_closed

        # Start the webview (this blocks until window is closed)
        # Use different backends based on platform for best compatibility
        logger.info("Starting webview window...")
        if sys.platform == "darwin":
            # macOS: use native WebKit
            webview.start(private_mode=False)
        elif sys.platform == "win32":
            # Windows: prefer EdgeChromium, fall back to others
            webview.start(private_mode=False)
        else:
            # Linux: use GTK WebKit
            webview.start(private_mode=False)

        logger.info("DocFinder closed normally.")

    except Exception as e:
        logger.exception("Fatal error during DocFinder startup: %s", e)
        # On Windows, show a message box so the user knows something went wrong
        if sys.platform == "win32":
            try:
                import ctypes

                log_path = _get_log_file_path()
                msg = f"DocFinder failed to start:\n\n{e}\n\nCheck the log file at:\n{log_path}"
                ctypes.windll.user32.MessageBoxW(0, msg, "DocFinder Error", 0x10)
            except Exception:
                pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()
