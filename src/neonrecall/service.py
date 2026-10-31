"""The NeonRecall facade: ingest, search, ask. Interfaces (API, CLI, MCP) wrap this."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from neonrecall.chunking import chunk_document
from neonrecall.config import Settings
from neonrecall.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from neonrecall.generation import Answerer, ExtractiveAnswerer, OpenAIChatAnswerer
from neonrecall.ingest import load_documents
from neonrecall.models import Answer, Document, ScoredChunk
from neonrecall.rerank import HTTPReranker, LexicalReranker, Reranker
from neonrecall.retrieval import HybridRetriever, Mode
from neonrecall.store.base import VectorStore
from neonrecall.store.sqlite import SQLiteStore


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"documents": self.documents, "chunks": self.chunks, "sources": self.sources}


def build_embedder(s: Settings) -> Embedder:
    if s.embed_backend == "hash":
        return HashingEmbedder(s.embed_dim or s.hash_dim)
    if s.embed_backend == "openai":
        return OpenAIEmbedder(
            s.embed_base_url, s.embed_model, s.embed_api_key or None, dim=s.embed_dim or None
        )
    raise ValueError(f"unknown embed_backend: {s.embed_backend!r}")


def build_store(s: Settings, dim: int) -> VectorStore:
    if s.store == "sqlite":
        return SQLiteStore(s.sqlite_path)
    if s.store == "postgres":
        from neonrecall.store.pgvector import PgVectorStore

        return PgVectorStore(s.pg_dsn, dim=dim, table=s.pg_table)
    raise ValueError(f"unknown store: {s.store!r}")


def build_reranker(s: Settings) -> Reranker | None:
    if s.rerank in ("", "none"):
        return None
    if s.rerank == "lexical":
        return LexicalReranker()
    if s.rerank == "http":
        return HTTPReranker(s.rerank_base_url, s.rerank_model, s.rerank_api_key or None)
    raise ValueError(f"unknown rerank: {s.rerank!r}")


def build_answerer(s: Settings) -> Answerer:
    if s.llm_backend == "extractive":
        return ExtractiveAnswerer()
    if s.llm_backend == "openai":
        return OpenAIChatAnswerer(s.llm_base_url, s.llm_model, s.llm_api_key or None)
    raise ValueError(f"unknown llm_backend: {s.llm_backend!r}")


class NeonRecall:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        answerer: Answerer | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store
        self.embedder = embedder
        self.answerer = answerer or ExtractiveAnswerer()
        self.retriever = HybridRetriever(
            store,
            embedder,
            reranker,
            candidates=self.settings.candidates,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> NeonRecall:
        s = settings or Settings.from_env()
        embedder = build_embedder(s)
        store = build_store(s, embedder.dim if s.store == "postgres" else 0)
        return cls(store, embedder, build_answerer(s), build_reranker(s), s)

    @classmethod
    def in_memory(cls, **overrides) -> NeonRecall:
        """Fully offline instance: SQLite in memory, hashing embedder, extractive answers."""
        s = Settings(
            **{
                "store": "sqlite",
                "sqlite_path": ":memory:",
                "embed_backend": "hash",
                "llm_backend": "extractive",
                **overrides,
            }
        )
        return cls.from_settings(s)

    # -- ingestion -----------------------------------------------------------------------

    def ingest_documents(self, docs: Iterable[Document]) -> IngestReport:
        report = IngestReport()
        for doc in docs:
            chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
            # Re-ingesting a source replaces it wholesale, so shrinking docs leave no orphans.
            self.store.delete_document(doc.doc_id)
            if chunks:
                vectors = self.embedder.embed([c.embedding_text() for c in chunks])
                self.store.upsert(chunks, vectors)
            report.documents += 1
            report.chunks += len(chunks)
            report.sources.append(doc.source)
        return report

    def ingest_paths(
        self, paths: Iterable[str | Path], base: str | Path | None = None
    ) -> IngestReport:
        return self.ingest_documents(load_documents(paths, base=base))

    # -- querying ------------------------------------------------------------------------

    def search(
        self, query: str, k: int | None = None, mode: Mode = "hybrid", rerank: bool | None = None
    ) -> list[ScoredChunk]:
        return self.retriever.retrieve(query, k or self.settings.top_k, mode=mode, rerank=rerank)

    def ask(
        self, question: str, k: int | None = None, mode: Mode = "hybrid", rerank: bool | None = None
    ) -> Answer:
        contexts = self.search(question, k, mode=mode, rerank=rerank)
        return self.answerer.answer(question, contexts)

    def stats(self) -> dict:
        return {
            "chunks": self.store.count(),
            "store": type(self.store).__name__,
            "embedder": self.embedder.name,
            "answerer": self.answerer.name,
            "reranker": self.retriever.reranker.name if self.retriever.reranker else None,
        }

    def close(self) -> None:
        self.store.close()
