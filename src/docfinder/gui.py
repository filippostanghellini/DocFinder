"""Desktop GUI launcher for DocFinder using pywebview.

This module provides a native desktop window that wraps the FastAPI web interface,
allowing users to interact with DocFinder through a graphical interface without
needing to use the command line.
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
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
    try:
        import webview
    except ImportError as exc:
        logger.error(
            "pywebview is not installed. "
            "Install the gui extras with: pip install 'docfinder[gui]'"
        )
        raise SystemExit(1) from exc

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
    # macOS uses the app bundle icon, Windows/Linux can use the icon parameter
    window_kwargs = {
        "title": "DocFinder",
        "url": url,
        "width": 1200,
        "height": 800,
        "min_size": (800, 600),
        "resizable": True,
        "text_select": True,
    }

    # Add icon on platforms that support it (not macOS - uses app bundle)
    if icon_path and sys.platform != "darwin":
        window_kwargs["icon"] = icon_path

    window = webview.create_window(**window_kwargs)

    def on_closed() -> None:
        """Handle window close event."""
        logger.info("Window closed, shutting down server...")
        server_thread.stop()

    window.events.closed += on_closed

    # Start the webview (this blocks until window is closed)
    # Use different backends based on platform for best compatibility
    if sys.platform == "darwin":
        # macOS: use native WebKit
        webview.start(private_mode=False)
    elif sys.platform == "win32":
        # Windows: prefer EdgeChromium, fall back to others
        webview.start(private_mode=False)
    else:
        # Linux: use GTK WebKit
        webview.start(private_mode=False)

    logger.info("DocFinder closed.")


if __name__ == "__main__":
    main()
