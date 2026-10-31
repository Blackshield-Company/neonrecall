"""Retrieval evaluation: recall@k and MRR over a labelled QA set."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from neonrecall.retrieval import Mode
from neonrecall.service import NeonRecall


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    relevant: tuple[str, ...]  # relevant document sources (corpus-relative paths)


@dataclass
class EvalResult:
    config: str
    k: int
    recall: float
    mrr: float
    hit_rate: float
    n: int
    misses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "k": self.k,
            f"recall@{self.k}": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            f"hit@{self.k}": round(self.hit_rate, 4),
            "n": self.n,
            "misses": self.misses,
        }


def load_qa(path: str | Path) -> list[QAItem]:
    items = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if not row.get("relevant"):
            raise ValueError(f"{path}:{line_no}: 'relevant' must be non-empty")
        items.append(QAItem(str(row["id"]), row["question"], tuple(row["relevant"])))
    return items


def ranked_sources(
    service: NeonRecall, question: str, k: int, mode: Mode, rerank: bool
) -> list[str]:
    """Distinct document sources in rank order, from the top-k chunks."""
    hits = service.search(question, k, mode=mode, rerank=rerank)
    return list(dict.fromkeys(h.chunk.source for h in hits))


def evaluate(
    service: NeonRecall,
    qa: Sequence[QAItem],
    k: int = 5,
    mode: Mode = "hybrid",
    rerank: bool = False,
) -> EvalResult:
    recall_sum = mrr_sum = hits = 0.0
    misses = []
    for item in qa:
        sources = ranked_sources(service, item.question, k, mode, rerank)
        relevant = set(item.relevant)
        found = relevant & set(sources)
        recall_sum += len(found) / len(relevant)
        rank = next((i for i, s in enumerate(sources, 1) if s in relevant), None)
        mrr_sum += 1.0 / rank if rank else 0.0
        if found:
            hits += 1
        else:
            misses.append(item.id)
    n = len(qa) or 1
    label = mode + ("+rerank" if rerank else "")
    return EvalResult(label, k, recall_sum / n, mrr_sum / n, hits / n, len(qa), misses)


CONFIGS: tuple[tuple[Mode, bool], ...] = (
    ("bm25", False),
    ("vector", False),
    ("hybrid", False),
    ("hybrid", True),
)


def run_suite(
    corpus: str | Path, qa_path: str | Path, k: int = 5, service: NeonRecall | None = None
) -> list[EvalResult]:
    """Ingest ``corpus`` into ``service`` (default: offline in-memory) and score each config."""
    owns = service is None
    if service is None:
        service = NeonRecall.in_memory(rerank="lexical")
    try:
        service.ingest_paths([corpus], base=corpus)
        qa = load_qa(qa_path)
        return [evaluate(service, qa, k, mode, rerank) for mode, rerank in CONFIGS]
    finally:
        if owns:
            service.close()


def format_table(results: Sequence[EvalResult]) -> str:
    if not results:
        return "(no results)"
    k = results[0].k
    header = f"{'config':<16}{'recall@' + str(k):>10}{'MRR':>8}{'hit@' + str(k):>8}{'n':>5}"
    lines = [header, "─" * len(header)]
    for r in results:
        lines.append(f"{r.config:<16}{r.recall:>10.3f}{r.mrr:>8.3f}{r.hit_rate:>8.3f}{r.n:>5}")
    return "\n".join(lines)
