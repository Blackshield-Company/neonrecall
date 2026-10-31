"""Command-line interface: ``neonrecall ingest|search|ask|stats|serve|mcp|eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neonrecall import __version__
from neonrecall.config import Settings
from neonrecall.retrieval import MODES

DEFAULT_CORPUS = Path("evals/corpus")
DEFAULT_QA = Path("evals/qa.jsonl")


def _service(args: argparse.Namespace):
    from neonrecall.service import NeonRecall

    return NeonRecall.from_settings(Settings.from_env(sqlite_path=args.db))


def _emit(data, as_json: bool, render) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False) if as_json else render(data))


def cmd_ingest(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        report = service.ingest_paths(args.paths, base=args.base).to_dict()
    finally:
        service.close()
    _emit(report, args.json, lambda r: f"ingested {r['documents']} docs → {r['chunks']} chunks")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        hits = [
            h.to_dict()
            for h in service.search(args.query, args.k, mode=args.mode, rerank=args.rerank or None)
        ]
    finally:
        service.close()

    def render(rows):
        out = []
        for i, h in enumerate(rows, 1):
            where = h["source"] + (
                f" › {h['metadata']['heading']}" if h["metadata"].get("heading") else ""
            )
            preview = " ".join(h["text"].split())[:160]
            out.append(f"{i:>2}. [{h['score']:.4f}] {where}\n    {preview}")
        return "\n".join(out) or "no results"

    _emit(hits, args.json, render)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        answer = service.ask(args.question, args.k, mode=args.mode, rerank=args.rerank or None)
    finally:
        service.close()

    def render(a):
        lines = [a["answer"], ""]
        lines += [
            f"  [{c['index']}] {c['source']}" + (f" › {c['heading']}" if c["heading"] else "")
            for c in a["citations"]
        ]
        return "\n".join(lines).rstrip()

    data = answer.to_dict()
    if not args.json:
        data.pop("contexts")
    _emit(data, args.json, render)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    service = _service(args)
    try:
        stats = service.stats()
    finally:
        service.close()
    _emit(stats, args.json, lambda s: "\n".join(f"{k:<10} {v}" for k, v in s.items()))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if args.db:
        os.environ["NEONRECALL_SQLITE_PATH"] = args.db
    uvicorn.run("neonrecall.api:app", host=args.host, port=args.port)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    import os

    from neonrecall.mcp_server import main as mcp_main

    if args.db:
        os.environ["NEONRECALL_SQLITE_PATH"] = args.db
    mcp_main(args.transport)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from neonrecall.evals import format_table, run_suite

    results = run_suite(args.corpus, args.qa, k=args.k)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(format_table(results))
    failed = [r for r in results if r.recall < args.min_recall or r.mrr < args.min_mrr]
    if failed:
        names = ", ".join(r.config for r in failed)
        print(
            f"FAIL: below thresholds (recall>={args.min_recall}, mrr>={args.min_mrr}): {names}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neonrecall", description="Hybrid-retrieval RAG with citations."
    )
    p.add_argument("--version", action="version", version=f"neonrecall {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite path (overrides NEONRECALL_SQLITE_PATH)")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="index Markdown/text files or directories", parents=[common])
    s.add_argument("paths", nargs="+")
    s.add_argument("--base", help="make sources relative to this directory")
    s.set_defaults(func=cmd_ingest)

    for name, func, arg in (("search", cmd_search, "query"), ("ask", cmd_ask, "question")):
        s = sub.add_parser(name, help=f"{name} the index", parents=[common])
        s.add_argument(arg)
        s.add_argument("-k", type=int, default=5)
        s.add_argument("--mode", choices=MODES, default="hybrid")
        s.add_argument("--rerank", action="store_true", help="force the configured reranker on")
        s.set_defaults(func=func)

    s = sub.add_parser("stats", help="show index and backend info", parents=[common])
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("serve", help="run the FastAPI server", parents=[common])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("mcp", help="run the MCP server", parents=[common])
    s.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    s.set_defaults(func=cmd_mcp)

    s = sub.add_parser("eval", help="run the offline retrieval eval", parents=[common])
    s.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    s.add_argument("--qa", type=Path, default=DEFAULT_QA)
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--min-recall", type=float, default=0.0)
    s.add_argument("--min-mrr", type=float, default=0.0)
    s.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
