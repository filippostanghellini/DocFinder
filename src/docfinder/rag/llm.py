"""Local LLM management via llama-cpp-python.

Handles model selection based on available system RAM, automatic download of
GGUF models from Hugging Face, and inference.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────────────
# Each entry maps a RAM tier to a recommended GGUF model.
# repo_id  : Hugging Face repository hosting the GGUF file
# filename : specific quantisation file to download
# ctx_size : default context window the model was trained with
# ram_min  : minimum available RAM in MB to use this tier


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """Specification for a single GGUF model variant."""

    name: str
    repo_id: str
    filename: str
    ctx_size: int
    ram_min_mb: int


# Ordered largest → smallest so the first match wins.
MODEL_TIERS: List[ModelSpec] = [
    ModelSpec(
        name="Qwen2.5-7B-Instruct",
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        ctx_size=8192,
        ram_min_mb=16_384,
    ),
    ModelSpec(
        name="Qwen2.5-3B-Instruct",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        ctx_size=8192,
        ram_min_mb=8_192,
    ),
    ModelSpec(
        name="Qwen2.5-1.5B-Instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        ctx_size=8192,
        ram_min_mb=0,  # fallback — runs on anything
    ),
]

_DEFAULT_MODELS_DIR = Path.home() / ".cache" / "docfinder" / "models"


def _get_total_ram_mb() -> int:
    """Return total physical RAM in MB using platform-native methods."""
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            import subprocess

            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            return total // (1024 * 1024)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024  # kB → MB
        except Exception:
            pass

    # Conservative fallback
    logger.warning("Unable to detect system RAM — defaulting to 8 GB assumption")
    return 8_192


def select_model(total_ram_mb: int | None = None) -> ModelSpec:
    """Pick the largest model that fits in available RAM."""
    if total_ram_mb is None:
        total_ram_mb = _get_total_ram_mb()

    for spec in MODEL_TIERS:
        if total_ram_mb >= spec.ram_min_mb:
            logger.info(
                "Selected model %s for %d MB RAM (requires >= %d MB)",
                spec.name,
                total_ram_mb,
                spec.ram_min_mb,
            )
            return spec

    # Should never happen since the last tier has ram_min_mb=0
    return MODEL_TIERS[-1]


def ensure_model(spec: ModelSpec, models_dir: Path | None = None) -> Path:
    """Download the GGUF file if not already cached and return its local path.

    Uses ``huggingface_hub`` (already a transitive dep of sentence-transformers)
    so we don't add new network dependencies.
    """
    dest_dir = models_dir or _DEFAULT_MODELS_DIR
    local_path = dest_dir / spec.filename

    if local_path.exists():
        logger.info("Model already cached at %s", local_path)
        return local_path

    logger.info("Downloading %s/%s …", spec.repo_id, spec.filename)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _detect_n_gpu_layers() -> int:
    """Return a sensible ``n_gpu_layers`` value.

    * Apple Silicon with Metal: offload all layers (``-1``).
    * CUDA available: offload all layers (``-1``).
    * Otherwise: CPU-only (``0``).
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return -1  # Metal
    try:
        import torch

        if torch.cuda.is_available():
            return -1
    except ImportError:
        pass
    return 0


class LocalLLM:
    """Wrapper around a ``llama_cpp.Llama`` instance."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int | None = None,
    ) -> None:
        from llama_cpp import Llama

        if n_gpu_layers is None:
            n_gpu_layers = _detect_n_gpu_layers()

        logger.info(
            "Loading LLM from %s  (n_ctx=%d, n_gpu_layers=%d)",
            model_path,
            n_ctx,
            n_gpu_layers,
        )
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        """Run a single completion and return the generated text."""
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        return output["choices"][0]["text"].strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Chat-completion style call (uses the model's chat template)."""
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["message"]["content"].strip()
