"""Embedding backends: OpenAI-compatible HTTP (OpenAI, Ollama, vLLM, LM Studio, ...)
and an offline feature-hashing embedder for tests and air-gapped use."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx

from neonrecall.text import tokenize

Vector = list[float]


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


class HashingEmbedder:
    """Deterministic bag-of-features embedder using the signed hashing trick.

    Features are word unigrams, word bigrams and character trigrams, so it captures
    lexical and some sub-word similarity. No model download, no network.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self.name = f"hashing-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> Counter[str]:
        words = tokenize(text)
        feats: Counter[str] = Counter()
        for i, word in enumerate(words):
            feats[f"w:{word}"] += 2
            if i + 1 < len(words):
                feats[f"b:{word}_{words[i + 1]}"] += 1
            padded = f"<{word}>"
            for j in range(len(padded) - 2):
                feats[f"c:{padded[j : j + 3]}"] += 0.5
        return feats

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self._dim
        for feat, weight in self._features(text).items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self._dim] += sign * (1.0 + math.log(weight))
        return normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbedder:
    """Calls ``POST {base_url}/embeddings`` on any OpenAI-compatible server.

    For Ollama use ``base_url="http://localhost:11434/v1"`` and e.g. ``nomic-embed-text``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        dim: int | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai:{model}"
        self.batch_size = batch_size
        self._dim = dim
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = headers

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                resp = self._client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise EmbeddingError(f"embedding request failed: {exc}") from exc
            if len(data) != len(batch):
                raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(data)}")
            out.extend(normalize(d["embedding"]) for d in data)
        if out and self._dim is None:
            self._dim = len(out[0])
        return out
