"""Extended tests for embedding backend to cover error handling paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docfinder.embedding import encoder


class TestCheckGPUAvailabilityExtended:
    """Extended tests for _check_gpu_availability."""

    def test_rocm_gpu_detection(self) -> None:
        """Detects AMD ROCm GPU via torch.version.hip."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.version.hip = "5.0"  # ROCm detected

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = encoder._check_gpu_availability()
            assert result == (True, "rocm")

    def test_gpu_detection_handles_exception(self) -> None:
        """Handles exception in GPU detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = encoder._check_gpu_availability()
            assert result == (False, None)


class TestCheckOnnxProvidersExtended:
    """Tests for _check_onnx_providers."""

    def test_returns_empty_on_import_error(self) -> None:
        """Returns empty list when onnxruntime not available."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            result = encoder._check_onnx_providers()
            assert result == []


class TestPreferredTorchDeviceExtended:
    """Tests for _preferred_torch_device."""

    def test_returns_cuda_for_rocm(self) -> None:
        """Returns cuda for ROCm GPU type."""
        with patch(
            "docfinder.embedding.encoder._check_gpu_availability",
            return_value=(True, "rocm"),
        ):
            result = encoder._preferred_torch_device()
            assert result == "cuda"

    def test_returns_cpu_no_gpu(self) -> None:
        """Returns cpu when no GPU available."""
        with patch(
            "docfinder.embedding.encoder._check_gpu_availability",
            return_value=(False, None),
        ):
            result = encoder._preferred_torch_device()
            assert result == "cpu"


class TestEmbeddingModelEmbed:
    """Tests for EmbeddingModel.embed methods."""

    def test_embed_with_custom_batch_size(self) -> None:
        """Uses custom batch_size when provided."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0] * 768]

        with patch.object(encoder.SentenceTransformer, "__init__", return_value=None):
            with patch.object(
                encoder.SentenceTransformer, "get_sentence_embedding_dimension", return_value=768
            ):
                config = encoder.EmbeddingConfig()
                model = encoder.EmbeddingModel(config)
                model._model = mock_model

                result = model.embed(["test"], batch_size=64)
                assert mock_model.encode.call_args[1]["batch_size"] == 64

    def test_embed_query_returns_single_embedding(self) -> None:
        """embed_query returns single embedding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[1.0] * 768]

        with patch.object(encoder.SentenceTransformer, "__init__", return_value=None):
            with patch.object(
                encoder.SentenceTransformer, "get_sentence_embedding_dimension", return_value=768
            ):
                config = encoder.EmbeddingConfig()
                model = encoder.EmbeddingModel(config)
                model._model = mock_model

                result = model.embed_query("test query")
                assert result.shape == (768,)
