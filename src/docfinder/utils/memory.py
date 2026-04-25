"""System memory utilities for adaptive resource management."""

from __future__ import annotations

import sys
from typing import Any


def get_memory_info() -> dict[str, Any]:
    """Return available and total RAM in MB using platform-native methods.

    Returns ``{"available_mb": int | None, "total_mb": int | None}``.
    No extra dependencies required.
    """
    try:
        import psutil  # optional – used if installed

        vm = psutil.virtual_memory()
        return {
            "available_mb": vm.available // (1024 * 1024),
            "total_mb": vm.total // (1024 * 1024),
        }
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            import subprocess as _sp

            total = int(_sp.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            vm_out = _sp.check_output(["vm_stat"]).decode()
            free_pages = inactive_pages = 0
            for line in vm_out.splitlines():
                if line.startswith("Pages free:"):
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive_pages = int(line.split(":")[1].strip().rstrip("."))
            available = (free_pages + inactive_pages) * 4096
            return {"available_mb": available // (1024 * 1024), "total_mb": total // (1024 * 1024)}
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo") as _f:
                for line in _f:
                    k, v = line.split(":")
                    meminfo[k.strip()] = int(v.strip().split()[0])  # kB
            return {
                "available_mb": meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) // 1024,
                "total_mb": meminfo.get("MemTotal", 0) // 1024,
            }
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import ctypes

            class _MemStatEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatEx()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return {
                "available_mb": stat.ullAvailPhys // (1024 * 1024),
                "total_mb": stat.ullTotalPhys // (1024 * 1024),
            }
        except Exception:
            pass

    return {"available_mb": None, "total_mb": None}


def compute_embed_batch_size(available_mb: int | None = None) -> tuple[int, float]:
    """Choose embedding batch size and inter-batch sleep based on available RAM.

    Args:
        available_mb: Available RAM in MB. If None, will be detected automatically.

    Returns:
        (batch_size, sleep_seconds) tuple.
    """
    if available_mb is None:
        available_mb = get_memory_info().get("available_mb")

    if available_mb is None or available_mb >= 4096:
        return 64, 0.0
    elif available_mb >= 2048:
        return 32, 0.0
    elif available_mb >= 1024:
        return 16, 0.0
    elif available_mb >= 512:
        return 8, 0.05
    else:
        return 4, 0.2
