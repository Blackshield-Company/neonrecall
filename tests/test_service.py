import pytest

from neonrecall import NeonRecall
from neonrecall.config import Settings
from neonrecall.models import Document


@pytest.mark.parametrize("failure", ["exception", "missing_vectors"])
def test_failed_embedding_preserves_existing_document(service, monkeypatch, failure):
    service.ingest_documents([Document("doc.md", "Original searchable content")])
    original = list(service.store.iter_chunks())
    revision = service.store.revision()

    def fail(texts):
        if failure == "exception":
            raise RuntimeError("embedding backend unavailable")
        return []

    monkeypatch.setattr(service.embedder, "embed", fail)
    with pytest.raises((RuntimeError, ValueError)):
        service.ingest_documents([Document("doc.md", "Replacement content")])
    assert list(service.store.iter_chunks()) == original
    assert service.store.revision() == revision


def test_reingest_replaces_old_chunks(service):
    long = "# Doc\n" + "\n\n".join(f"Paragraph {i} " + "filler " * 40 for i in range(10))
    first = service.ingest_documents([Document("doc.md", long)])
    assert first.chunks > 1
    service.ingest_documents([Document("doc.md", "# Doc\nnow tiny")])
    assert service.store.count() == 1


def test_ask_end_to_end(loaded):
    ans = loaded.ask("Which gas makes a sign glow blue?")
    assert "Argon" in ans.text
    assert ans.citations and ans.citations[0].source == "neon-sign-care.md"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("NEONRECALL_CHUNK_SIZE", "321")
    monkeypatch.setenv("NEONRECALL_BM25_WEIGHT", "0.5")
    monkeypatch.setenv("NEONRECALL_EMBED_BACKEND", "hash")
    s = Settings.from_env(sqlite_path=":memory:")
    assert s.chunk_size == 321 and s.bm25_weight == 0.5 and s.sqlite_path == ":memory:"


def test_stats(loaded):
    stats = loaded.stats()
    assert stats["chunks"] > 0
    assert stats["embedder"] == "hashing-512"
    assert stats["answerer"] == "extractive"


def test_sqlite_file_roundtrip(tmp_path):
    s = Settings(sqlite_path=str(tmp_path / "x.db"))
    svc = NeonRecall.from_settings(s)
    svc.ingest_documents([Document("a.md", "# A\nmagenta horizon grid")])
    svc.close()
    svc = NeonRecall.from_settings(s)
    assert svc.search("magenta horizon")[0].chunk.source == "a.md"
    svc.close()
