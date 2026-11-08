"""Utility helpers for working with files and modality detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".sh",
        ".sql",
        ".toml",
        ".ini",
    }
)

TEXT_EXTENSIONS: frozenset[str] = (
    frozenset(
        {
            ".pdf",
            ".txt",
            ".md",
            ".rst",
            ".tex",
            ".log",
            ".docx",
            ".odt",
            ".odp",
            ".odg",
            ".ods",
            ".doc",
            ".ppt",
            ".pptx",
            ".html",
            ".htm",
            ".epub",
            ".rtf",
            ".csv",
            ".tsv",
            ".xlsx",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".eml",
            ".mbox",
        }
    )
    | CODE_EXTENSIONS
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
    }
)

AUDIO_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
    }
)

SUPPORTED_EXTENSIONS: frozenset[str] = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_VIDEO_EXTENSIONS


def extensions_for_modalities(modalities: Sequence[str] | None = None) -> frozenset[str]:
    """Return the set of file extensions enabled by the active modalities.

    When ``modalities`` is None, all ``SUPPORTED_EXTENSIONS`` are returned.
    """
    if modalities is None:
        return SUPPORTED_EXTENSIONS
    mods = set(modalities)
    allowed = set(TEXT_EXTENSIONS)
    if "image" in mods:
        allowed |= IMAGE_EXTENSIONS
    if "audio" in mods:
        allowed |= AUDIO_VIDEO_EXTENSIONS
    return frozenset(allowed)


def iter_document_paths(
    inputs: Iterable[Path],
    *,
    modalities: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield supported document paths from input paths, descending into directories."""
    allowed = extensions_for_modalities(modalities)
    for item in inputs:
        if item.is_dir():
            yield from sorted(
                p
                for p in item.rglob("*")
                if (
                    p.is_file()
                    and not p.name.startswith("~$")
                    and not any(part.startswith(".") for part in p.relative_to(item).parts)
                    and p.suffix.lower() in allowed
                )
            )
        elif item.is_file() and not item.name.startswith("~$") and item.suffix.lower() in allowed:
            yield item


def _pdf_has_scanned_pages(path: Path, max_sample_pages: int = 3) -> bool:
    """Quickly inspect up to ``max_sample_pages`` of a PDF to detect image-only/scanned pages."""
    try:
        import fitz

        doc = fitz.open(path)
    except Exception:
        return False
    try:
        pages_to_check = min(len(doc), max_sample_pages)
        for idx in range(pages_to_check):
            page = doc[idx]
            text = (page.get_text() or "").strip()
            if len(text) < 30 and page.get_images():
                return True
        return False
    except Exception:
        return False
    finally:
        doc.close()


def detect_missing_modalities(
    files: Iterable[Path],
    active_modalities: Sequence[str],
    *,
    max_pdf_probes: int = 15,
) -> dict[str, dict[str, Any]]:
    """Inspect candidate files and report complex files requiring inactive modality encoders."""
    active = set(active_modalities or ("text",))
    need_image = "image" not in active
    need_audio = "audio" not in active
    if not need_image and not need_audio:
        return {}

    image_files: list[str] = []
    audio_files: list[str] = []
    pdf_probes = 0

    for f in files:
        ext = f.suffix.lower()
        if need_image and ext in IMAGE_EXTENSIONS:
            image_files.append(f.name)
        elif need_image and ext == ".pdf" and pdf_probes < max_pdf_probes:
            pdf_probes += 1
            if _pdf_has_scanned_pages(f):
                image_files.append(f"{f.name} (scanned PDF)")
        elif need_audio and ext in AUDIO_VIDEO_EXTENSIONS:
            audio_files.append(f.name)

    missing: dict[str, dict[str, Any]] = {}
    if image_files:
        missing["image"] = {
            "modality": "image",
            "label": "Scanned PDFs & Images (+Vision, +170M)",
            "count": len(image_files),
            "examples": image_files[:5],
            "ram_mb": 190,
            "description": (
                "Required to index standalone images and scanned PDF pages without selectable text."
            ),
        }
    if audio_files:
        missing["audio"] = {
            "modality": "audio",
            "label": "Audio & Video Recordings (+Audio, +300M)",
            "count": len(audio_files),
            "examples": audio_files[:5],
            "ram_mb": 310,
            "description": "Required to index voice notes, audio recordings, and video files.",
        }
    return missing


def iter_pdf_paths(inputs: Iterable[Path]) -> Iterator[Path]:
    """Yield PDF paths from input paths (kept for backward compatibility)."""
    for item in inputs:
        if item.is_dir():
            yield from sorted(p for p in item.rglob("*.pdf") if p.is_file())
        elif item.is_file() and item.suffix.lower() == ".pdf":
            yield item


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()
