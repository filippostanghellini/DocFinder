"""Tests for the Ollama client (urllib-based)."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docfinder.ollama import (
    OllamaEmbedder,
    OllamaError,
    OllamaLLM,
    _l2_normalize,
    _request,
    list_ollama_models,
)


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestRequest:
    """Tests for the urllib request helper."""

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_get_parses_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({"models": []})
        data = _request("http://localhost:11434", "/api/tags", timeout=5)
        assert data == {"models": []}

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_post_sends_payload(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({"ok": True})
        _request("http://localhost:11434/", "/api/chat", {"model": "x"})
        req = mock_urlopen.call_args[0][0]
        assert json.loads(req.data.decode()) == {"model": "x"}

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_bearer_header_only_with_api_key(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _fake_response({})
        _request("http://x", "/api/tags", api_key="secret")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret"

        _request("http://x", "/api/tags")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") is None

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_http_error_raises_ollama_error(self, mock_urlopen: MagicMock) -> None:
        err = urllib.error.HTTPError(
            "http://x", 404, "Not Found", {}, io.BytesIO(json.dumps({"error": "nope"}).encode())
        )
        mock_urlopen.side_effect = err
        with pytest.raises(OllamaError, match="HTTP 404"):
            _request("http://x", "/api/tags")

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_url_error_raises_unreachable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("refused")
        with pytest.raises(OllamaError, match="unreachable"):
            _request("http://x", "/api/tags")

    @patch("docfinder.ollama.urllib.request.urlopen")
    def test_bad_json_raises_ollama_error(self, mock_urlopen: MagicMock) -> None:
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        with pytest.raises(OllamaError, match="Invalid response"):
            _request("http://x", "/api/tags")


class TestListModels:
    @patch("docfinder.ollama._request")
    def test_returns_name_and_size(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "models": [{"name": "llama3", "size": 123}, {"name": "nomic-embed-text"}]
        }
        models = list_ollama_models("http://x")
        assert models == [
            {"name": "llama3", "size_bytes": 123},
            {"name": "nomic-embed-text", "size_bytes": None},
        ]


class TestL2Normalize:
    def test_normalizes_rows(self) -> None:
        vectors = np.array([[3.0, 0.0], [0.0, 2.0]], dtype="float32")
        out = _l2_normalize(vectors)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_zero_row_safe(self) -> None:
        vectors = np.array([[0.0, 0.0]], dtype="float32")
        assert np.allclose(_l2_normalize(vectors), 0.0)


class TestOllamaEmbedder:
    @patch("docfinder.ollama._request")
    def test_embed_returns_normalized_float32(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        embedder = OllamaEmbedder("http://x", "m")
        out = embedder.embed(["a", "b"])
        assert out.dtype == np.float32
        assert out.shape == (2, 2)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    @patch("docfinder.ollama._request")
    def test_invalid_response_raises(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"embeddings": []}
        with pytest.raises(OllamaError, match="no/invalid"):
            OllamaEmbedder("http://x", "m").embed(["a"])

    @patch("docfinder.ollama._request")
    def test_dimension_probed_and_memoized(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"embeddings": [[1.0, 2.0, 3.0]]}
        embedder = OllamaEmbedder("http://x", "m")
        assert embedder.dimension == 3
        assert embedder.dimension == 3
        assert mock_request.call_count == 1

    @patch("docfinder.ollama._request")
    def test_embed_query_single_vector(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"embeddings": [[1.0, 2.0]]}
        out = OllamaEmbedder("http://x", "m").embed_query("q")
        assert out.shape == (2,)


class TestOllamaLLM:
    @patch("docfinder.ollama._request")
    def test_chat_returns_content(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"message": {"content": " hello "}}
        answer = OllamaLLM("http://x", "m").chat([{"role": "user", "content": "hi"}])
        assert answer == "hello"

    @patch("docfinder.ollama._request")
    def test_chat_maps_options(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"message": {"content": ""}}
        OllamaLLM("http://x", "m").chat([{"role": "user", "content": "hi"}], max_tokens=64)
        payload = mock_request.call_args[0][2]
        assert payload["options"]["num_predict"] == 64
        assert payload["stream"] is False

    @patch("docfinder.ollama._request")
    def test_chat_missing_message(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {}
        assert OllamaLLM("http://x", "m").chat([{"role": "user", "content": "hi"}]) == ""


class TestIsLocalUrl:
    def test_localhost_variants(self) -> None:
        from docfinder.ollama import is_local_url

        assert is_local_url("http://127.0.0.1:11434")
        assert is_local_url("http://localhost:11434")
        assert is_local_url("http://LOCALHOST:11434")
        assert is_local_url("https://[::1]:11434")

    def test_remote(self) -> None:
        from docfinder.ollama import is_local_url

        assert not is_local_url("http://vps.example.com:11434")
        assert not is_local_url("http://8.8.8.8:11434")

    def test_garbage(self) -> None:
        from docfinder.ollama import is_local_url

        assert is_local_url("not a url") is False  # urlsplit("") → no hostname
        assert is_local_url("") is False
