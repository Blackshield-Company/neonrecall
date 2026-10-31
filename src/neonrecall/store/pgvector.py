"""PostgreSQL + pgvector backend (HNSW, cosine distance).

Requires the ``postgres`` extra: ``pip install 'neonrecall[postgres]'``.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from neonrecall.models import Chunk

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, table: str = "neonrecall_chunks") -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "PostgreSQL backend needs psycopg: pip install 'neonrecall[postgres]'"
            ) from exc
        if not _IDENT_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self.dim = dim
        self.table = table
        self._lock = threading.RLock()
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        self._init_schema()

    def _init_schema(self) -> None:
        t = self.table
        with self._lock, self._conn.transaction():
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dim}) NOT NULL
                )"""
            )
            self._conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_doc_id ON {t} (doc_id)")
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw ON {t} "
                "USING hnsw (embedding vector_cosine_ops)"
            )
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {t}_meta (key TEXT PRIMARY KEY, value BIGINT NOT NULL)"
            )
            self._conn.execute(
                f"INSERT INTO {t}_meta (key, value) VALUES ('revision', 0) ON CONFLICT DO NOTHING"
            )
            row = self._conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                (t,),
            ).fetchone()
        if row and row["atttypmod"] not in (-1, self.dim):
            raise ValueError(
                f"table {t} stores {row['atttypmod']}-dim vectors but the embedder "
                f"produces {self.dim}; use a different NEONRECALL_PG_TABLE"
            )

    def _bump(self) -> None:
        self._conn.execute(f"UPDATE {self.table}_meta SET value = value + 1 WHERE key = 'revision'")

    @staticmethod
    def _to_chunk(row: dict[str, Any]) -> Chunk:
        meta = row["metadata"]
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            source=row["source"],
            title=row["title"],
            ordinal=row["ordinal"],
            text=row["text"],
            metadata=meta if isinstance(meta, dict) else json.loads(meta),
        )

    def revision(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {self.table}_meta WHERE key = 'revision'"
            ).fetchone()
        return int(row["value"])

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = [
            (
                c.id,
                c.doc_id,
                c.source,
                c.title,
                c.ordinal,
                c.text,
                json.dumps(c.metadata),
                _vec_literal(e),
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table}
                    (id, doc_id, source, title, ordinal, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id, source = EXCLUDED.source,
                        title = EXCLUDED.title, ordinal = EXCLUDED.ordinal,
                        text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding""",
                rows,
            )
            self._bump()

    def delete_document(self, doc_id: str) -> int:
        with self._lock, self._conn.transaction():
            cur = self._conn.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))
            if cur.rowcount:
                self._bump()
            return cur.rowcount

    def search(self, embedding: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT id, doc_id, source, title, ordinal, text, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s::vector, id
                    LIMIT %s""",
                (_vec_literal(embedding), _vec_literal(embedding), k),
            ).fetchall()
        return [(self._to_chunk(r), float(r["score"])) for r in rows]

    def iter_chunks(self) -> Iterator[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, doc_id, source, title, ordinal, text, metadata "
                f"FROM {self.table} ORDER BY source, ordinal"
            ).fetchall()
        return (self._to_chunk(r) for r in rows)

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()["n"]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
