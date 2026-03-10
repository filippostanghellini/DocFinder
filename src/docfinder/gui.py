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


class GlobalHotkeyManager:
    """Registers and manages a system-wide keyboard shortcut via pynput.

    When the hotkey fires, the DocFinder window is brought to the front
    and the search input is focused.
    """

    def __init__(self) -> None:
        self.window: object | None = None  # set after create_window()
        self._listener = None

    def start(self, hotkey: str, enabled: bool = True) -> None:
        """Register the global hotkey. Replaces any previously registered one."""
        self.stop()
        if not enabled or not hotkey:
            return
        try:
            from pynput import keyboard  # type: ignore[import-untyped]

            self._listener = keyboard.GlobalHotKeys({hotkey: self._on_activate})
            self._listener.daemon = True
            self._listener.start()
            logger.info("Global hotkey registered: %s", hotkey)
        except ImportError:
            logger.warning("pynput is not installed — global hotkey unavailable")
        except Exception as exc:
            logger.warning("Could not register global hotkey '%s': %s", hotkey, exc)

    def _on_activate(self) -> None:
        logger.debug("Global hotkey activated — bringing DocFinder to front")
        try:
            if sys.platform == "darwin":
                try:
                    from AppKit import NSApplication  # type: ignore[import-untyped]

                    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                except ImportError:
                    pass
            if self.window:
                self.window.show()
                self.window.evaluate_js(
                    "document.querySelector('[data-tab=\"search\"]').click();"
                    "setTimeout(()=>document.getElementById('query').focus(),60);"
                )
        except Exception as exc:
            logger.warning("Failed to bring window to front: %s", exc)

    def stop(self) -> None:
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def reload(self, hotkey: str, enabled: bool = True) -> None:
        """Stop the current listener and register a new hotkey."""
        self.start(hotkey, enabled)


class DesktopApi:
    """Exposes native OS capabilities to the webview JS frontend.

    Available from JavaScript as window.pywebview.api.*
    """

    def __init__(self) -> None:
        self.window: object | None = None
        self._hotkey_manager: GlobalHotkeyManager | None = None

    def pick_folder(self) -> str | None:
        """Open the native folder picker and return the selected absolute path, or None."""
        if self.window is None:
            return None
        try:
            import webview

            result = self.window.create_file_dialog(webview.FOLDER_DIALOG, allow_multiple=False)
            return result[0] if result else None
        except Exception as exc:
            logger.warning("Folder picker dialog failed: %s", exc)
            return None

    def reload_hotkey(self) -> None:
        """Re-read settings from disk and apply the hotkey immediately."""
        if self._hotkey_manager is None:
            return
        from docfinder.settings import load_settings

        s = load_settings()
        self._hotkey_manager.reload(s.get("hotkey", ""), s.get("hotkey_enabled", True))


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
        from docfinder.settings import load_settings

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

        icon_path = _get_icon_path()

        # ── macOS dock icon (source runs only) ───────────────────────────────
        # When frozen (PyInstaller), the bundle already carries the .icns icon.
        # When running from source, set the dock icon via AppKit so the user
        # doesn't see the generic Python icon.
        if sys.platform == "darwin" and not getattr(sys, "frozen", False) and icon_path:
            try:
                from AppKit import NSApplication, NSImage  # type: ignore[import-untyped]

                ns_app = NSApplication.sharedApplication()
                ns_image = NSImage.alloc().initWithContentsOfFile_(icon_path)
                if ns_image:
                    ns_app.setApplicationIconImage_(ns_image)
                    logger.debug("macOS dock icon set via AppKit")
            except ImportError:
                logger.debug("pyobjc not available — run: pip install 'docfinder[gui]'")
            except Exception as exc:
                logger.debug("Could not set macOS dock icon: %s", exc)

        # ── Desktop API (folder picker + hotkey reload) ──────────────────────
        desktop_api = DesktopApi()
        hotkey_manager = GlobalHotkeyManager()
        desktop_api._hotkey_manager = hotkey_manager

        # ── Window creation ──────────────────────────────────────────────────
        window_kwargs: dict = {
            "title": "DocFinder",
            "url": url,
            "width": 1200,
            "height": 800,
            "min_size": (800, 600),
            "resizable": True,
            "text_select": True,
            "js_api": desktop_api,
        }

        # Add icon on Windows — guard against older pywebview builds that
        # don't expose the `icon` parameter (would raise TypeError otherwise)
        if icon_path and sys.platform == "win32":
            import inspect

            if "icon" in inspect.signature(webview.create_window).parameters:
                window_kwargs["icon"] = icon_path
            else:
                logger.debug("pywebview does not support 'icon' parameter, skipping")

        try:
            window = webview.create_window(**window_kwargs)
        except TypeError as exc:
            logger.warning("create_window() failed (%s), retrying without icon", exc)
            window_kwargs.pop("icon", None)
            window = webview.create_window(**window_kwargs)

        # Wire window references
        desktop_api.window = window
        hotkey_manager.window = window

        def on_closed() -> None:
            logger.info("Window closed, shutting down...")
            hotkey_manager.stop()
            server_thread.stop()

        window.events.closed += on_closed

        # ── Start global hotkey after webview is ready ───────────────────────
        def _on_webview_started() -> None:
            settings = load_settings()
            hotkey_manager.start(
                settings.get("hotkey", ""),
                settings.get("hotkey_enabled", True),
            )

        logger.info("Starting webview window...")
        webview.start(private_mode=False, func=_on_webview_started)

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
