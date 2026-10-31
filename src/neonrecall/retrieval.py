"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Literal

from neonrecall.bm25 import BM25Index
from neonrecall.embeddings import Embedder
from neonrecall.models import Chunk, ScoredChunk
from neonrecall.rerank import Reranker
from neonrecall.store.base import VectorStore

Mode = Literal["hybrid", "vector", "bm25"]
MODES: tuple[Mode, ...] = ("hybrid", "vector", "bm25")


def rrf_fuse(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or {}
    fused: dict[str, ScoredChunk] = {}
    for name, ranked in ranked_lists.items():
        w = weights.get(name, 1.0)
        for rank, (chunk, raw) in enumerate(ranked, start=1):
            entry = fused.setdefault(chunk.id, ScoredChunk(chunk, 0.0))
            entry.score += w / (k + rank)
            entry.signals[name] = raw
            entry.signals[f"{name}_rank"] = rank
    return sorted(fused.values(), key=lambda s: (-s.score, s.chunk.id))


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        candidates: int = 30,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.weights = {"vector": vector_weight, "bm25": bm25_weight}
        self._bm25: tuple[int, BM25Index] | None = None

    def bm25(self) -> BM25Index:
        rev = self.store.revision()
        if self._bm25 is None or self._bm25[0] != rev:
            self._bm25 = (rev, BM25Index(list(self.store.iter_chunks())))
        return self._bm25[1]

    def retrieve(
        self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool | None = None
    ) -> list[ScoredChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if k <= 0 or not query.strip():
            return []
        pool = max(k, self.candidates)
        lists: dict[str, list[tuple[Chunk, float]]] = {}
        if mode in ("hybrid", "vector"):
            [qvec] = self.embedder.embed([query])
            lists["vector"] = self.store.search(qvec, pool)
        if mode in ("hybrid", "bm25"):
            lists["bm25"] = self.bm25().search(query, pool)
        results = rrf_fuse(lists, self.weights, self.rrf_k)
        use_rerank = self.reranker is not None if rerank is None else rerank
        if use_rerank and self.reranker is not None:
            results = self.reranker.rerank(query, results[:pool])
        return results[:k]
