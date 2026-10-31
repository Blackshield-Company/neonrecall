"""Okapi BM25 over chunk text."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from neonrecall.models import Chunk
from neonrecall.text import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tfs: list[Counter[str]] = []
        self._lens: list[int] = []
        df: Counter[str] = Counter()
        for chunk in self.chunks:
            tokens = tokenize(chunk.embedding_text())
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lens.append(len(tokens))
            df.update(tf.keys())
        n = len(self.chunks)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms or not self.chunks:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lens[i] / (self._avgdl or 1.0))
            for term in terms:
                freq = tf.get(term)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(self.chunks[i], s) for s, i in scored[:k]]
