"""Core data types shared across the pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    title: str = ""

    @property
    def doc_id(self) -> str:
        return hashlib.sha1(self.source.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    source: str
    title: str
    ordinal: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))

    def embedding_text(self) -> str:
        """Text sent to the embedder: the chunk prefixed with where it lives."""
        context = " > ".join(p for p in (self.title, self.heading) if p)
        return f"{context}\n{self.text}" if context else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "ordinal": self.ordinal,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.chunk.to_dict(), "score": self.score, "signals": dict(self.signals)}


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    source: str
    title: str
    heading: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "heading": self.heading,
            "snippet": self.snippet,
        }


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    contexts: list[ScoredChunk]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": [c.to_dict() for c in self.contexts],
            "backend": self.backend,
        }
