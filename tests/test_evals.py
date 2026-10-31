from neonrecall.evals import QAItem, evaluate, format_table, load_qa, run_suite
from tests.conftest import CORPUS, QA


def test_qa_set_is_well_formed():
    qa = load_qa(QA)
    assert len(qa) >= 20
    sources = {p.name for p in CORPUS.glob("*.md")}
    assert all(set(item.relevant) <= sources for item in qa)


def test_metrics_math(loaded):
    qa = [
        QAItem("hit", "degaussing wand purity", ("crt-maintenance.md",)),
        QAItem("miss", "degaussing wand purity", ("neon-sign-care.md",)),
    ]
    res = evaluate(loaded, qa, k=1, mode="bm25")
    assert res.recall == 0.5 and res.mrr == 0.5 and res.misses == ["miss"]


def test_suite_meets_quality_bar():
    results = {r.config: r for r in run_suite(CORPUS, QA, k=5)}
    assert set(results) == {"bm25", "vector", "hybrid", "hybrid+rerank"}
    assert results["hybrid"].recall >= 0.9
    assert results["hybrid"].mrr >= 0.85
    assert results["hybrid+rerank"].recall >= 0.9
    assert "recall@5" in format_table(list(results.values()))
