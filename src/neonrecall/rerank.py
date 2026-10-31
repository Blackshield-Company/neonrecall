"""Optional second-stage rerankers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from neonrecall.models import ScoredChunk
from neonrecall.text import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class LexicalReranker:
    """Offline reranker: rewards chunks that cover more distinct query terms, and
    cover them close together, blended with the first-stage rank."""

    name = "lexical"

    def __init__(self, prior_weight: float = 0.35, window: int = 12) -> None:
        self.prior_weight = prior_weight
        self.window = window

    def _score(self, terms: set[str], text: str) -> float:
        tokens = tokenize(text)
        if not terms or not tokens:
            return 0.0
        coverage = len(terms & set(tokens)) / len(terms)
        # Best density of query terms in any sliding window.
        best = 0
        for start in range(0, max(1, len(tokens) - self.window + 1)):
            best = max(best, len(terms & set(tokens[start : start + self.window])))
        proximity = best / len(terms)
        return 0.7 * coverage + 0.3 * proximity

    def rerank(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        rescored = []
        for rank, item in enumerate(results):
            lexical = self._score(terms, item.chunk.embedding_text())
            prior = 1.0 / math.log2(rank + 2)
            score = (1 - self.prior_weight) * lexical + self.prior_weight * prior
            rescored.append(ScoredChunk(item.chunk, score, {**item.signals, "rerank": lexical}))
        rescored.sort(key=lambda r: -r.score)
        return rescored


class HTTPReranker:
    """Cross-encoder served behind a ``POST {base_url}/rerank`` endpoint using the
    Cohere/Jina/vLLM request shape (``query`` + ``documents`` -> ``results``)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"http:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def rerank(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return []
        resp = self._client.post(
            f"{self.base_url}/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": [r.chunk.text for r in results],
                "top_n": len(results),
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        scores = {int(r["index"]): float(r["relevance_score"]) for r in resp.json()["results"]}
        rescored = [
            ScoredChunk(
                r.chunk, scores.get(i, float("-inf")), {**r.signals, "rerank": scores.get(i, 0.0)}
            )
            for i, r in enumerate(results)
        ]
        rescored.sort(key=lambda r: -r.score)
        return rescored
