from __future__ import annotations

from pathlib import Path

import pytest

from neonrecall import NeonRecall

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "evals" / "corpus"
QA = ROOT / "evals" / "qa.jsonl"


@pytest.fixture
def service():
    svc = NeonRecall.in_memory()
    yield svc
    svc.close()


@pytest.fixture
def loaded(service):
    service.ingest_paths([CORPUS], base=CORPUS)
    return service
