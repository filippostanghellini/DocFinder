"""Embedding model management (powered by Google EmbeddingGemma 2)."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "google/embeddinggemma-2"

# Legacy embedding models from earlier DocFinder releases that can be safely purged from cache
LEGACY_HF_MODELS: tuple[str, ...] = (
    "BAAI/bge-m3",
    "sentence-transformers/all-mpnet-base-v2",
)

MODALITY_SPECS: dict[str, dict[str, Any]] = {
    "text": {
        "id": "text",
        "label": "Text, Office & Code (Base)",
        "params": "270M",
        "ram_mb": 250,
        "required": True,
        "description": (
            "Searches digital PDFs, Word, Excel, CSV, PowerPoint, OpenDocument, "
            "Markdown, HTML, EPUB, emails (.eml/.mbox), and source code files."
        ),
    },
    "image": {
        "id": "image",
        "label": "Scanned PDFs & Images (+Vision)",
        "params": "+170M",
        "ram_mb": 190,
        "required": False,
        "description": (
            "Unlocks visual search inside scanned PDF pages (invoices, receipts, "
            "contracts without selectable text), screenshots, diagrams, and image files "
            "(.png, .jpg, .webp, .tiff, .bmp) without needing external OCR."
        ),
    },
    "audio": {
        "id": "audio",
        "label": "Audio & Video Recordings (+Audio)",
        "params": "+300M",
        "ram_mb": 310,
        "required": False,
        "description": (
            "Unlocks semantic search across voice notes, call recordings, podcasts "
            "(.wav, .mp3, .m4a, .flac, .ogg) and video files (.mp4, .mov, .mkv)."
        ),
    },
}

logger = logging.getLogger(__name__)


def is_embeddinggemma_model(model_name: str) -> bool:
    """Return True if the model name refers to an EmbeddingGemma checkpoint."""
    lowered = (model_name or "").lower()
    return any(tok in lowered for tok in ("embeddinggemma", "embedding-gemma", "embedding_gemma"))


def ollama_tag_for_modalities(modalities: Sequence[str]) -> str:
    """Return the official Ollama tag for EmbeddingGemma 2 matching active modalities."""
    mods = set(modalities or ("text",))
    if "image" in mods and "audio" in mods:
        return "embeddinggemma-2:740m"
    if "audio" in mods:
        return "embeddinggemma-2:570m"
    if "image" in mods:
        return "embeddinggemma-2:440m"
    return "embeddinggemma-2:270m"


def build_modality_config_kwargs(
    model_name: str, modalities: Sequence[str]
) -> dict[str, Any] | None:
    """Build SentenceTransformer config_kwargs to selectively load modality encoders."""
    if not is_embeddinggemma_model(model_name):
        return None
    mods = set(modalities or ("text",))
    config_kwargs: dict[str, Any] = {}
    if "image" not in mods:
        config_kwargs["vision_config"] = None
    if "audio" not in mods:
        config_kwargs["audio_config"] = None
    return config_kwargs or None


def format_document_text(
    text: str,
    *,
    title: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Format document chunk text with EmbeddingGemma 2's title/text instruction prefix."""
    if not is_embeddinggemma_model(model_name):
        return text
    # Avoid double-prefixing if already formatted
    if text.startswith("title: ") and " | text: " in text[:200]:
        return text
    clean_title = (title or "").strip() or "none"
    return f"title: {clean_title} | text: {text}"


def format_query_text(
    query: str,
    *,
    task: Literal["search", "code", "qa"] = "search",
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Format a user query with EmbeddingGemma 2's asymmetric task instruction prefix."""
    if not is_embeddinggemma_model(model_name):
        return query
    if query.startswith("task: ") and " | query: " in query[:100]:
        return query
    prefix_map = {
        "search": "search result",
        "code": "code retrieval",
        "qa": "question answering",
    }
    task_label = prefix_map.get(task, "search result")
    return f"task: {task_label} | query: {query}"


def cleanup_legacy_models(
    extra_models: Sequence[str] = (),
    cache_dir: Path | None = None,
) -> list[str]:
    """Remove unused legacy embedding models from the local Hugging Face hub cache.

    Returns a list of removed model directory names.
    """
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = Path(hf_home) / "hub"
        else:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        return []

    targets = list(LEGACY_HF_MODELS) + list(extra_models)
    removed: list[str] = []
    for repo_id in targets:
        folder_name = "models--" + repo_id.replace("/", "--")
        target_path = cache_dir / folder_name
        if target_path.exists() and target_path.is_dir():
            try:
                shutil.rmtree(target_path)
                removed.append(repo_id)
                logger.info("Removed legacy model cache: %s", target_path)
            except OSError as exc:
                logger.warning("Failed to remove legacy model cache %s: %s", target_path, exc)
    return removed


def is_hf_offline(timeout: float = 3.0) -> bool:
    """Detect whether Hugging Face Hub is unreachable (no internet).

    SentenceTransformer/CrossEncoder re-check huggingface.co on every load even
    when all model files are already in the local cache; without internet that
    turns into minutes of connection retries, freezing startup. When offline,
    callers must pass ``local_files_only=True`` so the cached copy loads directly.
    """
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return False
    except OSError:
        return True


def _check_gpu_availability() -> tuple[bool, str | None]:
    """Check if GPU is available and return GPU type.

    Returns:
        (has_gpu, gpu_type) where gpu_type is one of:
        - "cuda" for NVIDIA GPU
        - "rocm" for AMD GPU
        - "mps" for Apple Silicon GPU
        - None if no GPU available
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"CUDA GPU detected: {device_name}")
            return (True, "cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Apple MPS GPU detected")
            return (True, "mps")

        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.debug("AMD ROCm GPU detected")
            return (True, "rocm")

        logger.debug("No GPU detected, will use CPU")
        return (False, None)
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
        return (False, None)
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
        return (False, None)


def _check_onnx_providers() -> list[str]:
    """Check which ONNX Runtime execution providers are available."""
    try:
        import onnxruntime as ort

        return ort.get_available_providers()
    except ImportError:
        return []


def _preferred_torch_device() -> str:
    """Return the best torch device available on this host."""
    has_gpu, gpu_type = _check_gpu_availability()
    if has_gpu and gpu_type in {"cuda", "rocm"}:
        return "cuda"
    if has_gpu and gpu_type == "mps":
        return "mps"
    return "cpu"


def _preferred_torch_dtype(device: str | None = None) -> Any:
    """Return a safe torch dtype for EmbeddingGemma 2 (bfloat16 or float32, NEVER float16).

    EmbeddingGemma 2's activation range overflows float16 and silently produces NaNs.
    """
    try:
        import torch

        target = device or _preferred_torch_device()
        if target == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        if target == "mps":
            return torch.bfloat16
        return torch.float32
    except Exception:
        return None


def detect_optimal_backend() -> tuple[Literal["torch", "onnx"], str | None]:
    """Auto-detect the optimal backend based on hardware and platform."""
    backend, onnx_model_file, _device = detect_optimal_backend_config()
    return (backend, onnx_model_file)


def detect_optimal_backend_config() -> tuple[
    Literal["torch", "onnx"],
    str | None,
    str | None,
]:
    """Auto-detect backend and device with balanced best-available policy.

    Returns:
        (backend_name, onnx_model_file, device)
    """
    try:
        has_gpu, gpu_type = _check_gpu_availability()
        onnx_providers = _check_onnx_providers()

        # Apple MPS GPU - use PyTorch MPS
        if gpu_type == "mps":
            logger.info("Apple MPS GPU detected - using PyTorch MPS backend")
            return ("torch", None, "mps")

        # Apple Silicon without MPS - use ONNX if available, else PyTorch CPU
        if sys.platform == "darwin" and (
            platform.processor() == "arm" or platform.machine() == "arm64"
        ):
            if onnx_providers:
                logger.info("Detected Apple Silicon - using ONNX backend")
                return ("onnx", None, None)
            logger.info("Apple Silicon detected - ONNX unavailable, using PyTorch CPU")
            return ("torch", None, "cpu")

        # NVIDIA GPU
        if gpu_type == "cuda":
            if "CUDAExecutionProvider" in onnx_providers:
                logger.info("Detected NVIDIA GPU - using ONNX with CUDAExecutionProvider")
                return ("onnx", None, None)
            logger.info(
                "Detected NVIDIA GPU but ONNX CUDA provider unavailable - "
                "falling back to PyTorch CUDA"
            )
            return ("torch", None, "cuda")

        # AMD GPU (ROCm)
        if gpu_type == "rocm":
            if "ROCMExecutionProvider" in onnx_providers:
                logger.info("Detected AMD GPU - using ONNX with ROCMExecutionProvider")
                return ("onnx", None, None)
            logger.info(
                "Detected AMD GPU but ONNX ROCm provider unavailable - "
                "falling back to PyTorch ROCm device"
            )
            return ("torch", None, "cuda")

        # CPU only
        if onnx_providers:
            logger.info(
                "Detected platform %s - using ONNX backend (providers: %s)",
                sys.platform,
                ", ".join(onnx_providers),
            )
            return ("onnx", None, None)

        logger.info("ONNX not available - using PyTorch CPU backend on %s", sys.platform)
        return ("torch", None, "cpu" if not has_gpu else None)

    except Exception as e:
        logger.warning("Failed to detect optimal backend: %s, falling back to PyTorch CPU", e)
        return ("torch", None, "cpu")


def get_runtime_environment_info() -> dict[str, object]:
    """Return runtime environment information used for backend decisions."""
    has_gpu, gpu_type = _check_gpu_availability()
    onnx_providers = _check_onnx_providers()
    backend, onnx_model_file, device = detect_optimal_backend_config()
    return {
        "platform": sys.platform,
        "has_gpu": has_gpu,
        "gpu_type": gpu_type,
        "onnx_providers": onnx_providers,
        "selected_backend": backend,
        "selected_device": device,
        "selected_onnx_model": onnx_model_file,
    }


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    normalize: bool = True
    backend: Literal["torch", "onnx", "openvino"] | None = None
    onnx_model_file: str | None = None
    device: str | None = None
    modalities: tuple[str, ...] = ("text",)
    truncate_dim: Literal[128, 256, 512, 768] | None = None


class EmbeddingModel:
    """Wrapper around `SentenceTransformer` supporting EmbeddingGemma 2 multimodality.

    Features:
    - Selective modality encoder loading (Text 270M, +Vision 170M, +Audio 300M)
    - Safe numerical precision (`bfloat16` or `float32`, never `float16`)
    - Task instruction prefixes for asymmetric retrieval (`SearchQuery` / `Document`)
    - Matryoshka dimension truncation (`truncate_dim`)
    - Fallback to PyTorch if ONNX is unavailable for the checkpoint
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.config.modalities = tuple(self.config.modalities or ("text",))
        self.runtime_info: dict[str, object] = {
            "selected_backend": self.config.backend,
            "selected_device": self.config.device,
            "selected_onnx_model": self.config.onnx_model_file,
            "active_modalities": list(self.config.modalities),
        }

        if self.config.backend is None:
            # EmbeddingGemma 2 uses custom Gemma 4 multimodal modules in PyTorch
            if is_embeddinggemma_model(self.config.model_name):
                self.config.backend = "torch"
                if self.config.device is None:
                    self.config.device = _preferred_torch_device()
            else:
                backend, onnx_model_file, device = detect_optimal_backend_config()
                self.config.backend = backend
                self.config.onnx_model_file = onnx_model_file
                if self.config.device is None:
                    self.config.device = device

        self.runtime_info.update(get_runtime_environment_info())
        self.runtime_info["selected_backend"] = self.config.backend
        self.runtime_info["selected_device"] = self.config.device
        self.runtime_info["selected_onnx_model"] = self.config.onnx_model_file
        self.runtime_info["active_modalities"] = list(self.config.modalities)

        try:
            self._model = self._load_model()
            logger.info(
                "Successfully loaded embedding model %s (backend=%s, device=%s, modalities=%s)",
                self.config.model_name,
                self.config.backend,
                self.config.device or "auto",
                ",".join(self.config.modalities),
            )
        except Exception as e:
            if self.config.backend != "torch":
                logger.warning(
                    "Failed to load model with backend '%s': %s. Falling back to PyTorch.",
                    self.config.backend,
                    e,
                )
                self.config.backend = "torch"
                self.config.onnx_model_file = None
                if self.config.device is None:
                    self.config.device = _preferred_torch_device()
                self.runtime_info["selected_backend"] = self.config.backend
                self.runtime_info["selected_device"] = self.config.device
                self.runtime_info["selected_onnx_model"] = self.config.onnx_model_file
                self._model = self._load_model()
            else:
                raise

        # sentence-transformers >=6.1 renamed the method; fall back for older versions
        try:
            raw_dim = int(self._model.get_embedding_dimension())
        except AttributeError:
            raw_dim = int(self._model.get_sentence_embedding_dimension())
        self.dimension = int(self.config.truncate_dim or raw_dim)
        self._log_backend_info()

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model with backend, precision and modality settings."""
        model_kwargs: dict[str, Any] = {}

        if self.config.backend == "onnx" and self.config.onnx_model_file:
            model_kwargs["file_name"] = self.config.onnx_model_file

        if self.config.backend == "torch" and is_embeddinggemma_model(self.config.model_name):
            dtype = _preferred_torch_dtype(self.config.device)
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype

        config_kwargs = build_modality_config_kwargs(self.config.model_name, self.config.modalities)
        offline = is_hf_offline()

        kwargs: dict[str, Any] = {
            "backend": self.config.backend,
            "device": self.config.device,
            "local_files_only": offline,
            "model_kwargs": model_kwargs if model_kwargs else None,
        }
        if config_kwargs:
            kwargs["config_kwargs"] = config_kwargs
        if self.config.truncate_dim is not None:
            kwargs["truncate_dim"] = self.config.truncate_dim

        return SentenceTransformer(self.config.model_name, **kwargs)

    def _log_backend_info(self) -> None:
        """Log information about the backend being used."""
        info_parts = [f"Backend: {self.config.backend}"]

        if self.config.backend == "onnx":
            try:
                import onnxruntime as ort

                providers = ort.get_available_providers()
                info_parts.append(f"ONNX Providers: {', '.join(providers)}")
            except ImportError:
                logger.debug("ONNX Runtime not available for provider info")

            if self.config.onnx_model_file:
                info_parts.append(f"ONNX Model: {self.config.onnx_model_file}")

        if self.config.device:
            info_parts.append(f"Device: {self.config.device}")

        info_parts.append(f"Modalities: {','.join(self.config.modalities)}")
        selected = self.runtime_info.get("selected_backend")
        device = self.runtime_info.get("selected_device")
        if selected:
            info_parts.append(f"Selected: {selected}/{device or 'auto'}")

        logger.info(" | ".join(info_parts))

    def embed(
        self,
        texts: Sequence[Any] | Iterable[Any],
        *,
        batch_size: int | None = None,
        titles: Sequence[str | None] | None = None,
        is_Document: bool = False,
    ) -> np.ndarray:
        """Return float32 embeddings for text or multimodal inputs.

        Args:
            texts: Input strings or multimodal items (PIL.Image, dicts with
                ``{"text": ..., "image": ...}``, audio arrays/paths) to embed.
            batch_size: Override the configured batch size.
            titles: Optional per-item document titles for EmbeddingGemma 2 formatting.
            is_Document: When True and model is EmbeddingGemma 2, format plain text
                items with ``title: {title} | text: {text}``.
        """
        items = list(texts)
        if not items:
            return np.empty((0, self.dimension), dtype="float32")

        if is_embeddinggemma_model(self.config.model_name):
            formatted: list[Any] = []
            for idx, item in enumerate(items):
                title = titles[idx] if titles and idx < len(titles) else None
                if isinstance(item, str):
                    if item.startswith("task: ") and " | query: " in item[:100]:
                        formatted.append(item)
                    else:
                        formatted.append(
                            format_document_text(
                                item, title=title, model_name=self.config.model_name
                            )
                        )
                else:
                    # Multimodal inputs (images, audio, or interleaved dicts) are passed directly
                    formatted.append(item)
            items = formatted

        encode_kwargs: dict[str, Any] = {
            "batch_size": batch_size if batch_size is not None else self.config.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize,
        }
        if self.config.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.config.truncate_dim

        embeddings = self._model.encode(items, **encode_kwargs)
        return np.asarray(embeddings, dtype="float32")

    def embed_query(
        self,
        text: str,
        *,
        task: Literal["search", "code", "qa"] = "search",
    ) -> np.ndarray:
        """Generate a single query embedding using EmbeddingGemma 2's task instruction."""
        formatted = format_query_text(text, task=task, model_name=self.config.model_name)
        return self.embed([formatted])[0]
