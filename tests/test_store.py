import os

import pytest

from neonrecall.embeddings import HashingEmbedder
from neonrecall.models import Chunk
from neonrecall.store.sqlite import SQLiteStore


def _chunks(doc_id, texts):
    return [
        Chunk(f"{doc_id}:{i}", doc_id, f"{doc_id}.md", doc_id, i, t) for i, t in enumerate(texts)
    ]


def _exercise(store):
    emb = HashingEmbedder(dim=64)
    a = _chunks("a", ["synth filter resonance", "envelope attack decay"])
    b = _chunks("b", ["tape humidity storage"])
    rev0 = store.revision()
    store.upsert(a + b, emb.embed([c.text for c in a + b]))
    assert store.count() == 3
    assert store.revision() > rev0

    [q] = emb.embed(["tape humidity"])
    top_chunk, top_score = store.search(q, 2)[0]
    assert top_chunk.id == "b:0"
    assert 0 < top_score <= 1.0 + 1e-6

    # Upsert replaces in place.
    store.upsert([_chunks("a", ["replaced text"])[0]], emb.embed(["replaced text"]))
    assert store.count() == 3
    assert {c.text for c in store.iter_chunks()} >= {"replaced text"}

    assert store.delete_document("a") == 2
    assert store.delete_document("missing") == 0
    assert [c.id for c in store.iter_chunks()] == ["b:0"]

    with pytest.raises(ValueError):
        store.upsert(a, [])


def test_sqlite_memory_store():
    store = SQLiteStore(":memory:")
    _exercise(store)
    store.close()


def test_sqlite_persists_to_disk(tmp_path):
    path = str(tmp_path / "idx.db")
    store = SQLiteStore(path)
    store.upsert(_chunks("d", ["persist me"]), HashingEmbedder(16).embed(["persist me"]))
    store.close()
    reopened = SQLiteStore(path)
    assert reopened.count() == 1
    assert next(reopened.iter_chunks()).metadata == {}
    reopened.close()


@pytest.mark.postgres
@pytest.mark.skipif(
    not os.environ.get("NEONRECALL_TEST_PG_DSN"), reason="NEONRECALL_TEST_PG_DSN not set"
)
def test_pgvector_store():
    from neonrecall.store.pgvector import PgVectorStore

    store = PgVectorStore(os.environ["NEONRECALL_TEST_PG_DSN"], dim=64, table="neonrecall_test")
    try:
        for doc_id in {c.doc_id for c in store.iter_chunks()}:
            store.delete_document(doc_id)
        _exercise(store)
    finally:
        store._conn.execute("DROP TABLE IF EXISTS neonrecall_test, neonrecall_test_meta")
        store.close()
