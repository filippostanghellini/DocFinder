"""Check GitHub releases for DocFinder updates and install them in-app."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import docfinder

LOGGER = logging.getLogger("docfinder.updater")

GITHUB_REPO = os.environ.get("DOCFINDER_UPDATE_REPO", "filippostanghellini/DocFinder")
_TIMEOUT = 20.0

VERSION_RE = re.compile(r"v?(\d+(?:\.\d+)+)")


class UpdateCheckError(RuntimeError):
    """Raised when an update check or install cannot proceed."""


_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "state": "idle",
    "phase": None,
    "progress": 0.0,
    "downloaded": 0,
    "total": 0,
    "message": "",
    "error": None,
    "info": None,
}
UPDATE_DIR = Path(tempfile.gettempdir()) / "docfinder-update"


def _parse_version(text: str) -> tuple[int, ...] | None:
    match = VERSION_RE.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _version_key(value: tuple[int, ...] | None) -> tuple[int, int, int, int]:
    if not value:
        return (0, 0, 0, 0)
    padded = value + (0,) * max(0, 4 - len(value))
    return (padded[0], padded[1], padded[2], padded[3])


def _pick_asset(assets: list[dict]) -> dict | None:
    def _is_darwin(name: str) -> bool:
        return name.lower().endswith(".dmg")

    def _is_windows(name: str) -> bool:
        return name.lower().endswith(".exe") and "setup" in name.lower()

    def _is_linux(name: str) -> bool:
        return name.lower().endswith(".appimage")

    if sys.platform == "darwin":
        matcher = _is_darwin
    elif sys.platform == "win32":
        matcher = _is_windows
    else:
        matcher = _is_linux

    for asset in assets:
        name = str(asset.get("name") or "")
        if matcher(name):
            url = str(asset.get("browser_download_url") or "")
            if not url.startswith(f"https://github.com/{GITHUB_REPO}/releases/download/"):
                continue
            return {"name": name, "url": url, "size": int(asset.get("size") or 0)}
    return None


def _set_state(**kwargs: object) -> None:
    with _STATE_LOCK:
        _STATE.update(kwargs)


def status() -> dict:
    with _STATE_LOCK:
        return {key: value for key, value in _STATE.items()}


def check_for_update() -> dict:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "DocFinder-Updater",
            },
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise UpdateCheckError(f"Could not reach GitHub releases: {exc}") from exc

    if not isinstance(data, dict) or not data.get("tag_name"):
        raise UpdateCheckError("GitHub returned no release data")

    current = _parse_version(docfinder.__version__)
    latest = _parse_version(str(data.get("tag_name") or ""))
    available = bool(latest) and _version_key(latest) > _version_key(current)
    asset = _pick_asset(list(data.get("assets") or []))

    info = {
        "current_version": docfinder.__version__,
        "latest_version": ".".join(str(part) for part in latest)
        if latest
        else str(data.get("tag_name") or ""),
        "available": available,
        "release_url": str(data.get("html_url") or f"https://github.com/{GITHUB_REPO}/releases"),
        "release_notes": str(data.get("body") or ""),
        "asset": asset,
    }
    _set_state(info=info)
    return info


def _macos_app_dir() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.suffix == ".app":
            return parent
    return None


def _download(url: str, destination: Path, total: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "DocFinder"})
    with (
        urllib.request.urlopen(request, timeout=_TIMEOUT) as response,
        destination.open("wb") as output,
    ):
        total_size = total or int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            _set_state(
                downloaded=downloaded,
                total=total_size or _STATE.get("total") or downloaded,
                progress=min(1.0, downloaded / total_size) if total_size else 0.0,
            )


def _install_macos(installer: Path, info: dict) -> None:
    app_dir = _macos_app_dir()
    if app_dir is None:
        raise UpdateCheckError(
            "In-app update requires the packaged DocFinder.app. "
            "Download it manually from the release page."
        )
    parent = app_dir.parent
    if not os.access(parent, os.W_OK):
        raise UpdateCheckError(
            f"DocFinder.app at {app_dir} is not writable — move it to "
            "/Applications first, or update manually."
        )

    attach = subprocess.run(
        ["/usr/bin/hdiutil", "attach", "-nobrowse", "-readonly", str(installer)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if attach.returncode != 0:
        raise UpdateCheckError(f"hdiutil attach failed: {attach.stderr.strip()}")
    match = re.search(r"(/Volumes/.+)$", attach.stdout, re.MULTILINE)
    if match is None:
        raise UpdateCheckError("The DMG mounted but the volume path was not found.")
    mount_point = Path(match.group(1).strip())
    try:
        new_app = next(iter(mount_point.glob("*.app")), None)
        if new_app is None:
            raise UpdateCheckError("No .app bundle found inside the downloaded DMG.")
        temporary = parent / f"{app_dir.stem}.app.update"
        backup = parent / f"{app_dir.stem}.app.old"
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        subprocess.run(["/usr/bin/ditto", str(new_app), str(temporary)], check=True, timeout=600)
        subprocess.run(["/usr/bin/xattr", "-cr", str(temporary)], capture_output=True, timeout=60)
        os.replace(app_dir, backup)
        os.replace(temporary, app_dir)
    finally:
        subprocess.run(
            ["/usr/bin/hdiutil", "detach", str(mount_point)], capture_output=True, timeout=60
        )

    _set_state(state="restarting", message="Update installed — restarting DocFinder…")
    time.sleep(1.5)
    subprocess.Popen(["/usr/bin/open", str(app_dir)], start_new_session=True)
    threading.Thread(
        target=lambda: (time.sleep(3), shutil.rmtree(backup, ignore_errors=True)), daemon=True
    ).start()
    os.kill(os.getpid(), signal.SIGTERM)


def _install_windows(installer: Path, info: dict) -> None:
    if not getattr(sys, "frozen", False):
        raise UpdateCheckError(
            "In-app update works from the installed DocFinder build. "
            "Download the installer manually from the release page."
        )
    _set_state(state="restarting", message="Update installed — run the installer that appears…")
    time.sleep(0.5)
    subprocess.Popen([str(installer), "/S"], start_new_session=True)
    os.kill(os.getpid(), signal.SIGTERM)


def _install_linux(installer: Path, info: dict) -> None:
    current = os.environ.get("APPIMAGE")
    if not current and sys.argv and str(sys.argv[0]).endswith(".AppImage"):
        current = sys.argv[0]
    if not current:
        raise UpdateCheckError(
            "In-app update works when running as an AppImage. "
            "Download the new one manually from the release page."
        )
    target = Path(current).resolve()
    temporary = target.with_name(target.name + ".update")
    shutil.copyfile(installer, temporary)
    temporary.chmod(0o755)
    os.replace(temporary, target)
    _set_state(state="restarting", message="Update installed — restarting DocFinder…")
    time.sleep(1.2)
    subprocess.Popen([str(target)], start_new_session=True)
    os.kill(os.getpid(), signal.SIGTERM)


def _install_worker(info: dict) -> None:
    asset = info.get("asset") or {}
    url = str(asset.get("url") or "")
    name = str(asset.get("name") or "update")
    destination = UPDATE_DIR / name
    try:
        UPDATE_DIR.mkdir(parents=True, exist_ok=True)
        for existing in UPDATE_DIR.iterdir():
            if existing.is_file():
                existing.unlink(missing_ok=True)
        _set_state(downloaded=0, progress=0.0, message="Downloading update…")
        _download(url, destination, int(asset.get("size") or 0))
        _set_state(state="installing", phase="install", message="Installing update…")
        if sys.platform == "darwin":
            _install_macos(destination, info)
        elif sys.platform == "win32":
            _install_windows(destination, info)
        else:
            _install_linux(destination, info)
    except Exception as exc:
        LOGGER.warning("Update failed: %s", exc)
        _set_state(state="error", message=str(exc), error=str(exc))


def start_install() -> dict:
    with _STATE_LOCK:
        if _STATE["state"] in ("downloading", "installing", "restarting"):
            return {"status": "already-running", "state": _STATE["state"]}
        info = _STATE.get("info")
        if not (isinstance(info, dict) and info.get("available")):
            raise UpdateCheckError("No pending update — check for updates first.")
        if not (isinstance(info.get("asset"), dict) and info["asset"].get("url")):
            raise UpdateCheckError(
                "No installer available for this platform — download from the release page."
            )
        _STATE.update(
            {
                "state": "downloading",
                "progress": 0.0,
                "downloaded": 0,
                "total": info["asset"].get("size") or 0,
                "error": None,
                "message": "Downloading update…",
                "phase": "download",
            }
        )
    threading.Thread(
        target=_install_worker, args=(info,), daemon=True, name="docfinder-updater"
    ).start()
    return {"status": "started", "state": "downloading"}
