"""Settings, read from ``NEONRECALL_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

_PREFIX = "NEONRECALL_"


def _env(name: str, default: Any) -> Any:
    raw = os.environ.get(_PREFIX + name.upper())
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # Storage: "sqlite" (path may be ":memory:") or "postgres".
    store: str = "sqlite"
    sqlite_path: str = "neonrecall.db"
    pg_dsn: str = "postgresql://neonrecall:neonrecall@localhost:5432/neonrecall"
    pg_table: str = "neonrecall_chunks"

    # Chunking.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings: "hash" (offline) or "openai" (any OpenAI-compatible server).
    embed_backend: str = "hash"
    embed_base_url: str = "http://localhost:11434/v1"
    embed_model: str = "nomic-embed-text"
    embed_api_key: str = ""
    embed_dim: int = 0  # 0 = auto-detect for remote models; hashing uses 512
    hash_dim: int = 512

    # Generation: "extractive" (offline) or "openai".
    llm_backend: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.1"
    llm_api_key: str = ""

    # Retrieval.
    top_k: int = 5
    candidates: int = 30
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    # Reranking: "none", "lexical" (offline) or "http" (POST /rerank).
    rerank: str = "none"
    rerank_base_url: str = "http://localhost:8080/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_key: str = ""

    # API: /ingest with "paths" may only read below this directory.
    ingest_root: str = field(default_factory=os.getcwd)

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        values = {f.name: _env(f.name, getattr(cls(), f.name)) for f in fields(cls)}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
