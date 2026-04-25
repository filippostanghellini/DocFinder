"""Embedding model management."""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

logger = logging.getLogger(__name__)


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

        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"CUDA GPU detected: {device_name}")
            return (True, "cuda")

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Apple MPS GPU detected")
            return (True, "mps")

        # ROCm detection is trickier - check if CUDA backend is actually ROCm
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
    """Check which ONNX Runtime execution providers are available.

    Returns:
        List of available provider names.
    """
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


def detect_optimal_backend() -> tuple[Literal["torch", "onnx"], str | None]:
    """Auto-detect the optimal backend based on hardware and platform.

    Returns:
        A tuple of (backend_name, onnx_model_file).
        - backend_name: "torch" or "onnx"
        - onnx_model_file: ONNX model file name (if backend is "onnx"), or None

    Strategy:
        - Apple Silicon (M1/M2/M3): Use ONNX with ARM64 quantized model + CoreML
        - NVIDIA GPU (CUDA): Use ONNX with CUDAExecutionProvider if available
        - AMD GPU (ROCm): Use ONNX with ROCMExecutionProvider if available
        - Intel Mac: Use ONNX with standard model
        - CPU only: Use ONNX with standard model
    """
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

        # Apple Silicon - prefer quantized ONNX model
        if sys.platform == "darwin" and (
            platform.processor() == "arm" or platform.machine() == "arm64"
        ):
            logger.info("Detected Apple Silicon - using ONNX with ARM64 quantized model")
            return ("onnx", "onnx/model_qint8_arm64.onnx", None)

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

        # Apple MPS GPU on non-ONNX path
        if gpu_type == "mps":
            if onnx_providers:
                logger.info("Detected Apple MPS GPU - using ONNX backend")
                return ("onnx", None, None)
            logger.info("Detected Apple MPS GPU - ONNX unavailable, using PyTorch MPS")
            return ("torch", None, "mps")

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


class EmbeddingModel:
    """Thin wrapper around `SentenceTransformer` for query and document embeddings.

    Features:
    - Automatic backend detection (ONNX on macOS, PyTorch elsewhere)
    - Support for quantized ONNX models on Apple Silicon
    - Configurable device selection
    - Fallback to PyTorch if ONNX fails
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.runtime_info: dict[str, object] = {
            "selected_backend": self.config.backend,
            "selected_device": self.config.device,
            "selected_onnx_model": self.config.onnx_model_file,
        }

        # Auto-detect backend if not specified
        if self.config.backend is None:
            backend, onnx_model_file, device = detect_optimal_backend_config()
            self.config.backend = backend
            self.config.onnx_model_file = onnx_model_file
            if self.config.device is None:
                self.config.device = device

        self.runtime_info.update(get_runtime_environment_info())
        self.runtime_info["selected_backend"] = self.config.backend
        self.runtime_info["selected_device"] = self.config.device
        self.runtime_info["selected_onnx_model"] = self.config.onnx_model_file

        # Try to load model with specified backend, fallback to PyTorch on error
        try:
            self._model = self._load_model()
            logger.info(
                "Successfully loaded embedding model with backend=%s device=%s",
                self.config.backend,
                self.config.device or "auto",
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

        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._log_backend_info()

    def _load_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model with appropriate backend settings."""
        model_kwargs = {}

        # Add ONNX-specific model kwargs
        if self.config.backend == "onnx" and self.config.onnx_model_file:
            model_kwargs["file_name"] = self.config.onnx_model_file

        # Load model
        return SentenceTransformer(
            self.config.model_name,
            backend=self.config.backend,
            device=self.config.device,
            model_kwargs=model_kwargs if model_kwargs else None,
        )

    def _log_backend_info(self) -> None:
        """Log information about the backend being used."""
        info_parts = [f"Backend: {self.config.backend}"]

        if self.config.backend == "onnx":
            # Get ONNX Runtime providers if available
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

        selected = self.runtime_info.get("selected_backend")
        device = self.runtime_info.get("selected_device")
        if selected:
            info_parts.append(f"Selected: {selected}/{device or 'auto'}")

        logger.info(" | ".join(info_parts))

    def embed(
        self, texts: Sequence[str] | Iterable[str], *, batch_size: int | None = None
    ) -> np.ndarray:
        """Return float32 embeddings for input texts.

        Args:
            texts: Input strings to embed.
            batch_size: Override the configured batch size (useful for low-RAM scenarios).
        """
        sentences = list(texts)
        embeddings = self._model.encode(
            sentences,
            batch_size=batch_size if batch_size is not None else self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-query embedding."""
        return self.embed([text])[0]
