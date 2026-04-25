"""Cross-platform native desktop notifications. No extra dependencies."""

from __future__ import annotations

import logging
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def send_notification(title: str, message: str) -> None:
    """Send a native OS notification. Fails silently if unsupported."""
    try:
        if sys.platform == "darwin":
            _notify_macos(title, message)
        elif sys.platform.startswith("linux"):
            _notify_linux(title, message)
        elif sys.platform == "win32":
            _notify_windows(title, message)
        else:
            LOGGER.debug("Notifications not supported on %s", sys.platform)
    except Exception as exc:
        LOGGER.debug("Failed to send notification: %s", exc)


def _notify_macos(title: str, message: str) -> None:
    """Use osascript to display a macOS notification."""
    # Escape double quotes for AppleScript
    safe_title = title.replace('"', '\\"')
    safe_msg = message.replace('"', '\\"')
    script = f'display notification "{safe_msg}" with title "{safe_title}"'
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify_linux(title: str, message: str) -> None:
    """Use notify-send (libnotify) for Linux desktop notifications."""
    subprocess.Popen(
        ["notify-send", title, message],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify_windows(title: str, message: str) -> None:
    """Use PowerShell toast notification on Windows 10+."""
    # BalloonTipIcon via .NET — works without extra packages
    ps_script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] > $null; "
        "$template = [Windows.UI.Notifications.ToastNotificationManager]::"
        "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
        "$textNodes = $template.GetElementsByTagName('text'); "
        f"$textNodes.Item(0).AppendChild($template.CreateTextNode('{title}')) > $null; "
        f"$textNodes.Item(1).AppendChild($template.CreateTextNode('{message}')) > $null; "
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
        "[Windows.UI.Notifications.ToastNotificationManager]::"
        "CreateToastNotifier('DocFinder').Show($toast)"
    )
    subprocess.Popen(
        ["powershell", "-Command", ps_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
