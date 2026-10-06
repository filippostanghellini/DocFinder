"""Tests for desktop GUI shutdown behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_stopping_tap_stops_run_loop_when_disabling_tap_fails(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """\
        import sys
        import types

        class ObjCError(Exception):
            pass

        calls = []
        quartz = types.ModuleType("Quartz")
        def disable_tap(tap, enabled):
            calls.append("disable")
            raise ObjCError("already stopped")
        quartz.CGEventTapEnable = disable_tap
        quartz.CFRunLoopStop = lambda loop: calls.append("stop")
        objc = types.ModuleType("objc")
        objc.error = ObjCError
        sys.modules["Quartz"] = quartz
        sys.modules["objc"] = objc

        from docfinder.gui import GlobalHotkeyManager
        manager = GlobalHotkeyManager()
        manager._tap = object()
        manager._tap_source = object()
        manager._tap_run_loop = object()
        manager.stop()
        assert calls == ["disable", "stop"]
        assert manager._tap is None
        """
    )
    env = {**os.environ, "HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path)}
    subprocess.run([sys.executable, "-c", script], check=True, env=env, capture_output=True)
