import json

import httpx
import pytest

from neonrecall.bm25 import BM25Index
from neonrecall.models import Chunk, ScoredChunk
from neonrecall.rerank import HTTPReranker, LexicalReranker
from neonrecall.retrieval import rrf_fuse


def _c(i, text):
    return Chunk(str(i), "d", "d.md", "", i, text)


def test_bm25_prefers_rare_terms():
    idx = BM25Index(
        [
            _c(0, "the tape is stored"),
            _c(1, "the tape is baked at 54 degrees"),
            _c(2, "synth filter"),
        ]
    )
    hits = idx.search("baked tape", k=3)
    assert hits[0][0].id == "1"
    assert {c.id for c, _ in hits} == {"0", "1"}
    assert idx.search("zzz unknown") == []
    assert BM25Index([]).search("anything") == []


def test_rrf_fuses_and_records_signals():
    a, b, c = _c(0, "a"), _c(1, "b"), _c(2, "c")
    fused = rrf_fuse({"vector": [(a, 0.9), (b, 0.5)], "bm25": [(b, 7.0), (c, 3.0)]})
    assert fused[0].chunk.id == "1"  # appears in both lists
    assert fused[0].signals == {"vector": 0.5, "vector_rank": 2, "bm25": 7.0, "bm25_rank": 1}
    weighted = rrf_fuse({"vector": [(a, 1.0)], "bm25": [(c, 1.0)]}, {"vector": 0.1})
    assert weighted[0].chunk.id == "2"


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
def test_modes_find_the_right_doc(loaded, mode):
    hits = loaded.search("sticky-shed syndrome baking tapes", k=3, mode=mode)
    assert hits[0].chunk.source == "tape-archiving.md"


def test_hybrid_signals_present(loaded):
    [top] = loaded.search("ground loop hum", k=1)
    assert {"vector", "bm25"} <= top.signals.keys()


def test_bad_mode_and_empty_query(loaded):
    with pytest.raises(ValueError):
        loaded.search("x", mode="nope")
    assert loaded.search("   ") == []


def test_bm25_cache_invalidates_on_write(service):
    from neonrecall.models import Document

    service.ingest_documents([Document("a.md", "# A\nquasar beacon")])
    assert service.search("quasar", mode="bm25")
    service.ingest_documents([Document("b.md", "# B\nnebula lantern")])
    assert service.search("nebula", mode="bm25")[0].chunk.source == "b.md"


def test_lexical_reranker_promotes_full_coverage():
    results = [
        ScoredChunk(_c(0, "tape stored somewhere"), 1.0),
        ScoredChunk(_c(1, "bake sticky tape at 54 degrees"), 0.9),
    ]
    out = LexicalReranker().rerank("bake sticky tape", results)
    assert out[0].chunk.id == "1"
    assert "rerank" in out[0].signals


def test_reranker_toggle(service):
    from neonrecall import NeonRecall

    svc = NeonRecall.in_memory(rerank="lexical")
    from tests.conftest import CORPUS

    svc.ingest_paths([CORPUS], base=CORPUS)
    assert "rerank" in svc.search("hum", k=1)[0].signals
    assert "rerank" not in svc.search("hum", k=1, rerank=False)[0].signals
    svc.close()


def test_http_reranker():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    rr = HTTPReranker(
        "http://rr/v1", "bge", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    out = rr.rerank("q", [ScoredChunk(_c(0, "first"), 1.0), ScoredChunk(_c(1, "second"), 0.5)])
    assert [r.chunk.id for r in out] == ["1", "0"]
    assert seen == {"model": "bge", "query": "q", "documents": ["first", "second"], "top_n": 2}
    assert rr.rerank("q", []) == []
