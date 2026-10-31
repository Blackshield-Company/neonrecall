from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from neonrecall.models import Chunk


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks with their (unit-normalised) embeddings."""

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document; returns the number removed."""

    def search(self, embedding: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, best first."""

    def iter_chunks(self) -> Iterator[Chunk]: ...

    def count(self) -> int: ...

    def revision(self) -> int:
        """Monotonic counter bumped on every write; used to invalidate caches."""

    def close(self) -> None: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
