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


class SpotlightPanel:
    """Native macOS NSPanel + WKWebView for Spotlight-style search.

    Bypasses pywebview entirely for the floating search panel — pywebview 6.x
    crashes on macOS 15 (Sequoia) when creating a second borderless window.
    This class creates the NSPanel and WKWebView directly via pyobjc/AppKit.

    JS → Python communication uses WKScriptMessageHandler: spotlight.html calls
        window.webkit.messageHandlers.spotlight.postMessage({action:'hide'})
    which triggers orderOut_ on the panel.
    """

    def __init__(self, spotlight_url: str) -> None:
        self._url = spotlight_url
        self._panel: object | None = None
        self._webview: object | None = None
        self._handler: object | None = None

    def setup(self) -> None:
        """Create the NSPanel and WKWebView. Must be called from the Cocoa main thread."""
        try:
            self._build()
            # Register hide callback so spotlight.html can close the panel via HTTP
            from docfinder.web.app import register_spotlight_hide_callback

            register_spotlight_hide_callback(self.hide)
            logger.info("SpotlightPanel ready at %s", self._url)
        except Exception as exc:
            logger.warning("SpotlightPanel.setup() failed: %s", exc)
            self._panel = None

    def _build(self) -> None:
        import AppKit  # type: ignore[import-untyped]
        import WebKit  # type: ignore[import-untyped]

        rect = AppKit.NSMakeRect(0, 0, 660, 480)

        # ── WKWebView ─────────────────────────────────────────────────────────
        config = WebKit.WKWebViewConfiguration.alloc().init()
        webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(rect, config)
        webview.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        # Use a solid dark background on the WKWebView to avoid deprecated
        # transparent-background APIs (_setDrawsTransparentBackground: was the
        # culprit in the original pywebview crash).  The spotlight.html body
        # background is set to the same dark color so no white edges show.
        webview.setUnderPageBackgroundColor_(
            AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                22 / 255, 27 / 255, 42 / 255, 1.0
            )
        )

        # ── NSPanel ───────────────────────────────────────────────────────────
        # NSWindowStyleMaskNonactivatingPanel (128): the panel becomes key and
        # receives keyboard events WITHOUT activating the parent application.
        # This is how Spotlight/Alfred work — the previous active app stays
        # frontmost while the panel floats above and accepts typed input.
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(AppKit.NSPopUpMenuWindowLevel)
        panel.setMovableByWindowBackground_(True)
        panel.setHasShadow_(True)
        panel.setWorksWhenModal_(True)
        panel.setBecomesKeyOnlyIfNeeded_(False)
        panel.setBackgroundColor_(
            AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                22 / 255, 27 / 255, 42 / 255, 1.0
            )
        )
        panel.setContentView_(webview)

        self._panel = panel
        self._webview = webview

        # ── Auto-hide when panel loses key status ─────────────────────────────
        # Python callables are bridged as ObjC blocks by pyobjc 9+.
        def _on_resign_key(notification: object) -> None:
            try:
                from PyObjCTools import AppHelper  # type: ignore[import-untyped]

                AppHelper.callAfter(self._do_hide)
            except Exception:
                pass

        AppKit.NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
            AppKit.NSWindowDidResignKeyNotification, panel, None, _on_resign_key
        )

        # ── Load spotlight URL ─────────────────────────────────────────────────
        ns_url = AppKit.NSURL.URLWithString_(self._url)
        webview.loadRequest_(AppKit.NSURLRequest.requestWithURL_(ns_url))

    def is_visible(self) -> bool:
        """Return True if the panel is currently on screen."""
        try:
            return bool(self._panel is not None and self._panel.isVisible())
        except Exception:
            return False

    def show(self) -> None:
        """Show and focus the panel. Safe to call from any thread."""
        if self._panel is None:
            return
        try:
            from PyObjCTools import AppHelper  # type: ignore[import-untyped]

            AppHelper.callAfter(self._do_show)
        except Exception as exc:
            logger.warning("SpotlightPanel.show() failed: %s", exc)

    def _do_show(self) -> None:
        """Raise the panel.  Must run on the Cocoa main thread.

        The app is **not** activated — keyboard input is handled by the
        CGEventTap which intercepts all keystrokes while the panel is
        visible and injects them into the WKWebView via JavaScript.
        This gives true Spotlight / Alfred behaviour: the previously-active
        app keeps focus and the panel simply floats above everything.
        """
        if self._panel is None or self._webview is None:
            return
        try:
            self._panel.center()
            self._panel.orderFrontRegardless()

            # Clear previous results
            self._webview.evaluateJavaScript_completionHandler_(
                "(function(){"
                "var i=document.getElementById('spotlight-input');"
                "var r=document.getElementById('results');"
                "if(r)r.innerHTML='';"
                "if(i){i.value='';}"
                "})();",
                None,
            )
        except Exception as exc:
            logger.warning("SpotlightPanel._do_show() failed: %s", exc)

    def forward_key(self, keycode: int, chars: str) -> None:
        """Inject a keystroke into the WKWebView.  Called from the main thread
        by the CGEventTap when the panel is visible."""
        if self._webview is None:
            return

        _ESCAPE = 53
        _BACKSPACE = 51
        _RETURN = 36
        _ARROW_UP = 126
        _ARROW_DOWN = 125

        if keycode == _ESCAPE:
            self._do_hide()
        elif keycode == _ARROW_DOWN:
            self._webview.evaluateJavaScript_completionHandler_(
                "document.dispatchEvent(new KeyboardEvent("
                "'keydown',{key:'ArrowDown',bubbles:true}));",
                None,
            )
        elif keycode == _ARROW_UP:
            self._webview.evaluateJavaScript_completionHandler_(
                "document.dispatchEvent(new KeyboardEvent("
                "'keydown',{key:'ArrowUp',bubbles:true}));",
                None,
            )
        elif keycode == _RETURN:
            self._webview.evaluateJavaScript_completionHandler_(
                "document.dispatchEvent(new KeyboardEvent("
                "'keydown',{key:'Enter',bubbles:true}));",
                None,
            )
        elif keycode == _BACKSPACE:
            self._webview.evaluateJavaScript_completionHandler_(
                "var i=document.getElementById('spotlight-input');"
                "i.value=i.value.slice(0,-1);"
                "i.dispatchEvent(new Event('input'));",
                None,
            )
        elif chars:
            safe = (
                chars.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "")
                .replace("\r", "")
            )
            if safe:
                self._webview.evaluateJavaScript_completionHandler_(
                    f"var i=document.getElementById('spotlight-input');"
                    f"i.value+='{safe}';"
                    f"i.dispatchEvent(new Event('input'));",
                    None,
                )

    def hide(self) -> None:
        """Hide the panel and deactivate the app. Safe to call from any thread."""
        if self._panel is None:
            return
        try:
            from PyObjCTools import AppHelper  # type: ignore[import-untyped]

            AppHelper.callAfter(self._do_hide)
        except Exception as exc:
            logger.warning("SpotlightPanel.hide() failed: %s", exc)

    def _do_hide(self) -> None:
        """Hide the panel.  The app was never activated so the previously
        focused app keeps focus automatically."""
        if self._panel is None:
            return
        try:
            self._panel.orderOut_(None)
        except Exception as exc:
            logger.warning("SpotlightPanel._do_hide() failed: %s", exc)


def _parse_pynput_hotkey(hotkey_str: str) -> tuple[int, int] | None:
    """Parse a pynput-style hotkey string into (modifier_flags, keycode).

    E.g. ``"<cmd>+<shift>+d"`` → ``(kCGEventFlagMaskCommand | kCGEventFlagMaskShift, 2)``.

    Returns ``None`` if the string cannot be parsed.
    """
    # Mapping of pynput modifier names to CGEvent modifier flag bits
    _MOD_MAP: dict[str, int] = {
        "cmd": 1 << 20,   # kCGEventFlagMaskCommand = 0x00100000
        "shift": 1 << 17,  # kCGEventFlagMaskShift   = 0x00020000
        "alt": 1 << 19,   # kCGEventFlagMaskAlternate = 0x00080000
        "ctrl": 1 << 18,  # kCGEventFlagMaskControl = 0x00040000
    }

    # macOS virtual keycodes for A-Z (lowercase key name → keycode)
    _KEY_MAP: dict[str, int] = {
        "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3, "g": 5,
        "h": 4, "i": 34, "j": 38, "k": 40, "l": 37, "m": 46, "n": 45,
        "o": 31, "p": 35, "q": 12, "r": 15, "s": 1, "t": 17, "u": 32,
        "v": 9, "w": 13, "x": 7, "y": 16, "z": 6,
        "space": 49, "f1": 122, "f2": 120, "f3": 99, "f4": 118,
        "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
        "f10": 109, "f11": 103, "f12": 111,
    }

    # macOS Option+key produces dead-key characters — map them back to base keys.
    # e.g. Option+D → ∂, Option+S → ß, Option+F → ƒ, etc.
    _DEADKEY_MAP: dict[str, str] = {
        "å": "a", "∫": "b", "ç": "c", "∂": "d", "´": "e", "ƒ": "f", "©": "g",
        "˙": "h", "ˆ": "i", "∆": "j", "˚": "k", "¬": "l", "µ": "m", "˜": "n",
        "ø": "o", "π": "p", "œ": "q", "®": "r", "ß": "s", "†": "t", "¨": "u",
        "√": "v", "∑": "w", "≈": "x", "¥": "y", "ω": "z",
    }

    parts = [p.strip().lower().strip("<>") for p in hotkey_str.split("+")]
    if not parts:
        return None

    flags = 0
    keycode: int | None = None
    for p in parts:
        if p in _MOD_MAP:
            flags |= _MOD_MAP[p]
        elif p in _KEY_MAP:
            keycode = _KEY_MAP[p]
        elif p in _DEADKEY_MAP and _DEADKEY_MAP[p] in _KEY_MAP:
            keycode = _KEY_MAP[_DEADKEY_MAP[p]]
        else:
            logger.warning("Unknown hotkey component: %r", p)
            return None

    if keycode is None:
        return None
    return (flags, keycode)


class GlobalHotkeyManager:
    """Registers and manages a system-wide keyboard shortcut.

    On macOS, uses a CGEventTap to intercept AND suppress the hotkey event so
    it never reaches Finder or other apps.  On other platforms, falls back to
    pynput (passive listener).

    When triggered, shows the SpotlightPanel (macOS) or an in-app overlay
    (other platforms / fallback).
    """

    def __init__(self) -> None:
        self.main_window: object | None = None
        self.spotlight_panel: SpotlightPanel | None = None
        self._listener = None
        self._tap = None  # CGEventTap handle (macOS)
        self._tap_source = None  # CFRunLoopSource (macOS)
        self._tap_thread: threading.Thread | None = None

    def start(self, hotkey: str, enabled: bool = True) -> None:
        """Register the global hotkey. Replaces any previously registered one."""
        self.stop()
        if not enabled or not hotkey:
            return

        if sys.platform == "darwin":
            self._start_cgeventtap(hotkey)
        else:
            self._start_pynput(hotkey)

    def _start_cgeventtap(self, hotkey: str) -> None:
        """Register a CGEventTap that intercepts and suppresses the hotkey."""
        parsed = _parse_pynput_hotkey(hotkey)
        if parsed is None:
            logger.warning("Could not parse hotkey for CGEventTap: %r", hotkey)
            self._start_pynput(hotkey)
            return

        target_flags, target_keycode = parsed

        try:
            import Quartz  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("Quartz not available, falling back to pynput")
            self._start_pynput(hotkey)
            return

        # Mask for the modifier flags we care about (ignore caps lock, etc.)
        MOD_MASK = (
            (1 << 20)  # Command
            | (1 << 19)  # Option
            | (1 << 18)  # Control
            | (1 << 17)  # Shift
        )

        # macOS keycode → character (US keyboard layout, unshifted / shifted)
        _KC_CHARS: dict[int, tuple[str, str]] = {
            0: ("a", "A"), 1: ("s", "S"), 2: ("d", "D"), 3: ("f", "F"),
            4: ("h", "H"), 5: ("g", "G"), 6: ("z", "Z"), 7: ("x", "X"),
            8: ("c", "C"), 9: ("v", "V"), 11: ("b", "B"), 12: ("q", "Q"),
            13: ("w", "W"), 14: ("e", "E"), 15: ("r", "R"), 16: ("y", "Y"),
            17: ("t", "T"), 18: ("1", "!"), 19: ("2", "@"), 20: ("3", "#"),
            21: ("4", "$"), 22: ("6", "^"), 23: ("5", "%"), 24: ("=", "+"),
            25: ("9", "("), 26: ("7", "&"), 27: ("-", "_"), 28: ("8", "*"),
            29: ("0", ")"), 30: ("]", "}"), 31: ("o", "O"), 32: ("u", "U"),
            33: ("[", "{"), 34: ("i", "I"), 35: ("p", "P"), 37: ("l", "L"),
            38: ("j", "J"), 39: ("'", '"'), 40: ("k", "K"), 41: (";", ":"),
            42: ("\\", "|"), 43: (",", "<"), 44: ("/", "?"), 45: ("n", "N"),
            46: ("m", "M"), 47: (".", ">"), 49: (" ", " "), 50: ("`", "~"),
        }
        _SHIFT_FLAG = 1 << 17

        def _tap_callback(proxy: object, event_type: int, event: object, refcon: object) -> object:
            """CGEventTap callback — runs on the tap thread's CFRunLoop.

            When the spotlight panel is NOT visible, only the hotkey is
            intercepted.  When it IS visible, **all** keystrokes are captured,
            forwarded to the WKWebView via JS injection, and suppressed so
            they never reach the previously-focused app.
            """
            try:
                if event_type == Quartz.kCGEventKeyDown:
                    keycode = Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventKeycode
                    )
                    flags = Quartz.CGEventGetFlags(event) & MOD_MASK

                    # ── Hotkey toggle ──────────────────────────────────────
                    if keycode == target_keycode and flags == target_flags:
                        logger.debug("CGEventTap: hotkey intercepted (keycode=%d)", keycode)
                        try:
                            from PyObjCTools import AppHelper  # type: ignore[import-untyped]

                            AppHelper.callAfter(self._on_activate)
                        except Exception:
                            self._on_activate()
                        return None

                    # ── Forward all keys while spotlight is visible ────────
                    if self.spotlight_panel and self.spotlight_panel.is_visible():
                        # Skip if Cmd or Ctrl are held (allow system shortcuts)
                        cmd_ctrl = (1 << 20) | (1 << 18)
                        if flags & cmd_ctrl:
                            return event

                        # Resolve the typed character from the keycode table.
                        # This avoids calling AppKit from the tap thread which
                        # would crash (NSEvent.eventWithCGEvent_ is not
                        # thread-safe).
                        pair = _KC_CHARS.get(keycode)
                        shifted = bool(flags & _SHIFT_FLAG)
                        chars = pair[1 if shifted else 0] if pair else ""

                        try:
                            from PyObjCTools import AppHelper  # type: ignore[import-untyped]

                            AppHelper.callAfter(
                                self.spotlight_panel.forward_key, keycode, chars
                            )
                        except Exception:
                            pass
                        return None  # suppress

                elif event_type == Quartz.kCGEventTapDisabledByTimeout:
                    Quartz.CGEventTapEnable(self._tap, True)
            except Exception as exc:
                logger.debug("CGEventTap callback error: %s", exc)
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,  # Active tap — can suppress events
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
            _tap_callback,
            None,
        )

        if tap is None:
            logger.warning(
                "CGEventTapCreate returned None — Accessibility permission may be required. "
                "Grant access in System Settings → Privacy & Security → Accessibility."
            )
            self._start_pynput(hotkey)
            return

        self._tap = tap
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._tap_source = source

        def _run_tap() -> None:
            loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(loop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            logger.info("CGEventTap hotkey active: %s", hotkey)
            Quartz.CFRunLoopRun()

        t = threading.Thread(target=_run_tap, daemon=True, name="CGEventTap")
        t.start()
        self._tap_thread = t
        logger.info("Global hotkey registered via CGEventTap: %s", hotkey)

    def _start_pynput(self, hotkey: str) -> None:
        """Fallback: register via pynput (passive listener, cannot suppress events)."""
        try:
            from pynput import keyboard  # type: ignore[import-untyped]

            self._listener = keyboard.GlobalHotKeys({hotkey: self._on_activate})
            self._listener.daemon = True
            self._listener.start()
            logger.info("Global hotkey registered via pynput: %s", hotkey)
        except ImportError:
            logger.warning("pynput is not installed — global hotkey unavailable")
        except Exception as exc:
            logger.warning("Could not register global hotkey '%s': %s", hotkey, exc)

    def _on_activate(self) -> None:
        """Toggle the spotlight panel or activate the main window overlay."""
        logger.debug("Global hotkey fired")

        if self.spotlight_panel is not None:
            if self.spotlight_panel.is_visible():
                self.spotlight_panel.hide()
            else:
                self.spotlight_panel.show()
        else:
            # Fallback: activate main window + show CSS overlay
            if sys.platform == "darwin":
                try:
                    import AppKit  # type: ignore[import-untyped]

                    AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                except Exception as exc:
                    logger.debug("AppKit activate failed: %s", exc)
            try:
                if self.main_window:
                    self.main_window.evaluate_js("showSpotlightOverlay();")
            except Exception as exc:
                logger.warning("Failed to show spotlight overlay: %s", exc)

    def stop(self) -> None:
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        if self._tap is not None:
            try:
                import Quartz  # type: ignore[import-untyped]

                Quartz.CGEventTapEnable(self._tap, False)
                if self._tap_source is not None:
                    # Stop the run loop so the thread exits
                    Quartz.CFRunLoopStop(
                        Quartz.CFRunLoopGetMain()
                    )
            except Exception:
                pass
            self._tap = None
            self._tap_source = None
            self._tap_thread = None

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
        self._spotlight_panel: SpotlightPanel | None = None

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

    def show_spotlight(self) -> None:
        """Show the spotlight panel (callable from in-app JS)."""
        if self._hotkey_manager:
            self._hotkey_manager._on_activate()

    def hide_spotlight(self) -> None:
        """Hide the spotlight panel (pywebview fallback for non-macOS or older builds)."""
        if self._spotlight_panel:
            self._spotlight_panel.hide()

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

        # ── Desktop API (folder picker + hotkey + spotlight) ─────────────────
        desktop_api = DesktopApi()
        hotkey_manager = GlobalHotkeyManager()
        desktop_api._hotkey_manager = hotkey_manager

        # ── Main window ──────────────────────────────────────────────────────
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

        # ── Spotlight panel (macOS: native NSPanel; other platforms: fallback) ─
        spotlight_panel: SpotlightPanel | None = None
        if sys.platform == "darwin":
            spotlight_panel = SpotlightPanel(f"{url}/spotlight")

        # Wire references
        desktop_api.window = window
        desktop_api._spotlight_panel = spotlight_panel
        hotkey_manager.main_window = window
        hotkey_manager.spotlight_panel = spotlight_panel

        def on_closed() -> None:
            logger.info("Window closed, shutting down...")
            hotkey_manager.stop()
            server_thread.stop()

        window.events.closed += on_closed

        # ── Start global hotkey after webview is ready ────────────────────────
        # NOTE: webview.start(func=...) runs func in a background thread, NOT on
        # the Cocoa main thread.  NSPanel/WKWebView creation MUST happen on the
        # main thread, so we dispatch setup() via AppHelper.callAfter.
        def _on_webview_started() -> None:
            settings = load_settings()
            hotkey_manager.start(
                settings.get("hotkey", ""),
                settings.get("hotkey_enabled", True),
            )
            if spotlight_panel is not None:
                try:
                    from PyObjCTools import AppHelper  # type: ignore[import-untyped]

                    AppHelper.callAfter(spotlight_panel.setup)
                except Exception as exc:
                    logger.warning("Could not schedule SpotlightPanel.setup: %s", exc)

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
