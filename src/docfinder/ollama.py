"""Ollama HTTP client and swappable embedding/LLM backends."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np


class OllamaError(Exception):
    """Raised when an Ollama server cannot be reached or returns an error."""


def _request(
    base_url: str, path: str, payload: dict | None = None, *, api_key: str = "", timeout: int = 60
) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("error", str(exc))
        except Exception:
            detail = str(exc)
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"Ollama server unreachable at {base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Invalid response from Ollama: {exc}") from exc


def list_ollama_models(base_url: str, *, api_key: str = "", timeout: int = 5) -> list[dict]:
    """Return models installed on the Ollama server: [{"name", "size_bytes"}]."""
    data = _request(base_url, "/api/tags", api_key=api_key, timeout=timeout)
    return [{"name": m["name"], "size_bytes": m.get("size")} for m in data.get("models", [])]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class OllamaEmbedder:
    """Drop-in for EmbeddingModel backed by an Ollama server (duck-typed)."""

    def __init__(self, base_url: str, model_name: str, *, api_key: str = "", timeout: int = 120):
        self.base_url = base_url
        self.model_name = model_name
        self.backend = "ollama"
        self.config = SimpleNamespace(model_name=model_name, backend="ollama", device=None)
        self._api_key = api_key
        self._timeout = timeout
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vectors = self.ollama_embed(["dimension probe"])
            self._dimension = int(vectors.shape[1])
        return self._dimension

    def ollama_embed(self, texts: list[str]) -> np.ndarray:
        data = _request(
            self.base_url,
            "/api/embed",
            {"model": self.model_name, "input": list(texts)},
            api_key=self._api_key,
            timeout=self._timeout,
        )
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError(f"Ollama returned no/invalid embeddings for {self.model_name}")
        return _l2_normalize(np.asarray(vectors, dtype="float32"))

    def embed(self, texts, *, batch_size=None) -> np.ndarray:
        return self.ollama_embed(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self.ollama_embed([text])[0]


class OllamaLLM:
    """Drop-in for LocalLLM backed by POST /api/chat (duck-typed)."""

    n_ctx = 8192  # ponytail: constant context budget; use /api/show if precision ever matters

    def __init__(self, base_url: str, model_name: str, *, api_key: str = "", timeout: int = 300):
        self.base_url = base_url
        self.model_name = model_name
        self._api_key = api_key
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        data = _request(
            self.base_url,
            "/api/chat",
            {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            api_key=self._api_key,
            timeout=self._timeout,
        )
        return (data.get("message") or {}).get("content", "").strip()
