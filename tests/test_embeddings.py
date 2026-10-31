import json
import math

import httpx
import pytest

from neonrecall.embeddings import EmbeddingError, HashingEmbedder, OpenAIEmbedder


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=False))


def test_hashing_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=256)
    a1, a2 = emb.embed(["neon grid at midnight", "neon grid at midnight"])
    assert a1 == a2
    assert len(a1) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)


def test_hashing_similarity_tracks_overlap():
    emb = HashingEmbedder()
    q, near, far = emb.embed(
        [
            "degauss the CRT monitor",
            "use a degaussing wand on the CRT",
            "store tape at low humidity",
        ]
    )
    assert _cos(q, near) > _cos(q, far)


def test_hashing_empty_text():
    [v] = HashingEmbedder(dim=8).embed([""])
    assert v == [0.0] * 8


def _mock_embeddings(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, request.headers.get("authorization"), body))
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0, 0.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_embedder_batches_orders_and_normalizes():
    calls = []
    emb = OpenAIEmbedder(
        "http://ollama:11434/v1/",
        "nomic-embed-text",
        "sk-test",
        batch_size=2,
        client=_mock_embeddings(calls),
    )
    vecs = emb.embed(["a", "bbb", "cc"])
    assert len(calls) == 2
    assert calls[0][0] == "/v1/embeddings"
    assert calls[0][1] == "Bearer sk-test"
    assert calls[0][2] == {"model": "nomic-embed-text", "input": ["a", "bbb"]}
    # Results are re-ordered by "index" and unit-normalised.
    assert vecs[1][0] > vecs[2][0] > vecs[0][0]
    assert all(math.isclose(_cos(v, v), 1.0) for v in vecs)
    assert emb.dim == 3


def test_openai_embedder_errors_are_wrapped():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(EmbeddingError):
        OpenAIEmbedder("http://x/v1", "m", client=client).embed(["hi"])
