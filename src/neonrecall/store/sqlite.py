"""SQLite backend with brute-force cosine search. Use ``":memory:"`` for tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from collections.abc import Iterator, Sequence

from neonrecall.models import Chunk
from neonrecall.store.base import cosine

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO meta(key, value) VALUES ('revision', '0');
"""


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        source=row["source"],
        title=row["title"],
        ordinal=row["ordinal"],
        text=row["text"],
        metadata=json.loads(row["metadata"]),
    )


class SQLiteStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._cache: tuple[int, list[tuple[Chunk, array]]] | None = None
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def _bump(self) -> None:
        self._conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    def revision(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
        return int(row[0])

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
                array("f", e).tobytes(),
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            self._bump()

    def delete_document(self, doc_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            if cur.rowcount:
                self._bump()
            return cur.rowcount

    def _vectors(self) -> list[tuple[Chunk, array]]:
        rev = self.revision()
        if self._cache is None or self._cache[0] != rev:
            with self._lock:
                rows = self._conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
            loaded = []
            for row in rows:
                vec = array("f")
                vec.frombytes(row["embedding"])
                loaded.append((_row_to_chunk(row), vec))
            self._cache = (rev, loaded)
        return self._cache[1]

    def search(self, embedding: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        scored = [(chunk, cosine(embedding, vec)) for chunk, vec in self._vectors()]
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]

    def iter_chunks(self) -> Iterator[Chunk]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM chunks ORDER BY source, ordinal").fetchall()
        return (_row_to_chunk(r) for r in rows)

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
