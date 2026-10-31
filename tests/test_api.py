import pytest
from fastapi.testclient import TestClient

from neonrecall.api import create_app
from tests.conftest import CORPUS, ROOT


@pytest.fixture
def client(service):
    service.settings.ingest_root = str(ROOT)
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 0


def test_ingest_documents_and_query(client):
    r = client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "source": "grid.md",
                    "text": "# Grid\nThe horizon grid is magenta and scrolls forever.",
                }
            ]
        },
    )
    assert r.status_code == 200 and r.json()["documents"] == 1

    r = client.post("/query", json={"query": "What color is the horizon grid?"})
    body = r.json()
    assert "magenta" in body["answer"]
    assert body["citations"][0]["source"] == "grid.md"


def test_ingest_paths_and_search_only(client):
    r = client.post("/ingest", json={"paths": ["evals/corpus"]})
    assert r.status_code == 200
    assert r.json()["documents"] == len(list(CORPUS.glob("*.md")))
    r = client.post(
        "/query", json={"query": "MIDI clock pulses", "k": 2, "answer": False, "mode": "bm25"}
    )
    results = r.json()["results"]
    assert len(results) == 2 and results[0]["source"] == "evals/corpus/midi-sync.md"


@pytest.mark.parametrize(
    "payload,status",
    [
        ({}, 422),
        ({"paths": ["../../etc"]}, 400),
        ({"paths": ["does/not/exist"]}, 404),
        ({"paths": ["pyproject.toml"]}, 400),
    ],
)
def test_ingest_rejects_bad_input(client, payload, status):
    assert client.post("/ingest", json=payload).status_code == status


def test_query_validation(client):
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/query", json={"query": "x", "k": 0}).status_code == 422
