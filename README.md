# NEONRECALL

```
  ░▒▓  N E O N R E C A L L  ▓▒░
  ─────────────────────────────
   ╲    ╲    ╲   │   ╱    ╱    ╱
    ╲    ╲    ╲  │  ╱    ╱    ╱
  ───────────────┼───────────────
```

**Retrieval-augmented answers from your own docs, with citations.**
Drop in Markdown, ask questions, get answers that point back at the source.
It runs on a local model at 2 AM with the Wi-Fi off, or on a hosted API.

- **Ingest** Markdown and plain text, split into heading-aware chunks with overlap
- **Embed** with any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio), or the built-in offline hashing embedder
- **Store** vectors in PostgreSQL + pgvector (HNSW), or SQLite / in-memory
- **Retrieve** with hybrid search: dense vectors + BM25, fused with Reciprocal Rank Fusion, plus optional reranking
- **Answer** with numbered `[n]` citations through any OpenAI-compatible chat API, or an offline extractive fallback
- **Serve** over FastAPI, a CLI, and an MCP server (`search_docs`, `ask`)
- **Measure** retrieval with a recall@k / MRR eval harness on a checked-in QA set

Everything has an offline default, so the tests and evals need no network, no GPU and no API keys.

---

## ▸ Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ,postgres for the pgvector backend

neonrecall ingest evals/corpus --base evals/corpus
neonrecall ask "Which gas makes a neon sign glow blue?"
```

```
Argon with a drop of mercury glows blue. [1]

  [1] neon-sign-care.md › Neon Sign Care > How the colors work
```

With no configuration this runs fully offline: SQLite file `neonrecall.db`, the hashing
embedder, and extractive answers.

## ▸ Plugging in real models

Neonrecall talks to anything that speaks the OpenAI `/v1/embeddings` and
`/v1/chat/completions` APIs. With [Ollama](https://ollama.com):

```bash
docker compose up -d db                       # Postgres 16 + pgvector
docker compose --profile ollama up -d         # optional: Ollama in a container
ollama pull nomic-embed-text && ollama pull llama3.1

cp .env.example .env && set -a && . ./.env && set +a
neonrecall ingest docs/
neonrecall ask "How do we archive the tape masters?"
```

Settings are read from `NEONRECALL_*` environment variables (see `src/neonrecall/config.py`):

| Variable | Default | Notes |
|---|---|---|
| `NEONRECALL_STORE` | `sqlite` | `sqlite` or `postgres` |
| `NEONRECALL_SQLITE_PATH` | `neonrecall.db` | `:memory:` for throwaway runs |
| `NEONRECALL_PG_DSN` | `postgresql://neonrecall:neonrecall@localhost:5432/neonrecall` | matches `docker-compose.yml` |
| `NEONRECALL_PG_TABLE` | `neonrecall_chunks` | use one table per embedding model |
| `NEONRECALL_EMBED_BACKEND` | `hash` | `hash` or `openai` |
| `NEONRECALL_EMBED_BASE_URL` / `_MODEL` / `_API_KEY` | Ollama, `nomic-embed-text` | |
| `NEONRECALL_LLM_BACKEND` | `extractive` | `extractive` or `openai` |
| `NEONRECALL_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | Ollama, `llama3.1` | |
| `NEONRECALL_RERANK` | `none` | `none`, `lexical` (offline) or `http` (`POST /rerank`, Cohere/Jina/vLLM shape) |
| `NEONRECALL_CHUNK_SIZE` / `_CHUNK_OVERLAP` | `800` / `120` | characters |
| `NEONRECALL_VECTOR_WEIGHT` / `_BM25_WEIGHT` | `1.0` / `1.0` | RRF weights |
| `NEONRECALL_INGEST_ROOT` | current dir | `/ingest` path requests must stay inside it |

> Switching embedding models changes vector dimensions. Re-ingest into a fresh SQLite
> file or a new `NEONRECALL_PG_TABLE`; the pgvector store refuses mismatched dimensions.

## ▸ Interfaces

### CLI

```bash
neonrecall ingest PATH... [--base DIR]
neonrecall search "ground loop hum" -k 5 --mode hybrid|vector|bm25 [--rerank]
neonrecall ask "How do I stop the 60 Hz hum?" [--json]
neonrecall stats
neonrecall serve --port 8000          # FastAPI
neonrecall mcp                        # MCP over stdio
neonrecall eval --min-recall 0.9      # retrieval eval; non-zero exit below threshold
```

`--db PATH` overrides the SQLite file for any command.

### HTTP API

```bash
neonrecall serve
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"documents": [{"source": "grid.md", "text": "# Grid\nThe horizon grid is magenta."}]}'
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "What color is the grid?", "k": 5}'
```

| Endpoint | Body | Returns |
|---|---|---|
| `POST /ingest` | `{"documents": [{source, text, title?}], "paths": ["docs/"]}` | counts and sources |
| `POST /query` | `{query, k=5, mode="hybrid", rerank?, answer=true}` | answer, citations, contexts (or `results` when `answer=false`) |
| `GET /health` | | status and backend info |

Interactive docs live at `/docs`.

### MCP

Two tools, `search_docs(query, k, mode)` and `ask(question, k)`. Example client config:

```json
{
  "mcpServers": {
    "neonrecall": {
      "command": "/path/to/neonrecall/.venv/bin/neonrecall",
      "args": ["mcp", "--db", "/path/to/neonrecall.db"]
    }
  }
}
```

## ▸ How retrieval works

```
 query ─┬─► embed ──► vector top-N (cosine) ─┐
        │                                    ├─► RRF fuse ─► [rerank] ─► top-k ─► answer + [n] cites
        └─► BM25 top-N ──────────────────────┘
```

1. **Chunking.** Markdown is split at headings (code fences are respected), and each chunk
   keeps a breadcrumb such as `Tape Archiving > Sticky-shed syndrome`. Sections are packed
   paragraph by paragraph, then sentence by sentence, into ~800-character chunks that
   overlap by ~120 characters on word boundaries. The breadcrumb is prepended at embed time.
2. **Hybrid search.** Dense and BM25 candidates are fused with Reciprocal Rank Fusion, so
   no score calibration is needed. Exact terms like part numbers, error codes and `24 PPQN`
   are caught by BM25; paraphrases are caught by the vectors.
3. **Reranking (optional).** `lexical` scores query-term coverage and proximity offline;
   `http` calls a cross-encoder such as `bge-reranker` behind a `/rerank` endpoint.
4. **Answering.** Passages are numbered and the model is told to cite them. Citations are
   parsed back out of the text and mapped to source, heading and snippet. The offline
   extractive answerer picks the most query-relevant sentences and cites each one.

The BM25 index is built in-process from the store and rebuilt when the store's revision
counter changes. That suits corpora up to tens of thousands of chunks. Beyond that, move
keyword search into Postgres.

## ▸ Evals

`evals/corpus/` holds a small fictional night-shift studio handbook (CRTs, synths, tape,
neon). `evals/qa.jsonl` holds 28 questions labelled with their relevant documents.

```bash
neonrecall eval -k 5
```

```
config            recall@5     MRR   hit@5    n
───────────────────────────────────────────────
bm25                 1.000   1.000   1.000   28
vector               0.982   0.964   1.000   28
hybrid               1.000   1.000   1.000   28
hybrid+rerank        1.000   1.000   1.000   28
```

These numbers use the offline hashing embedder. CI fails if recall@5 drops below 0.9 or MRR
below 0.85. The set is small on purpose: it's a regression tripwire, not a benchmark. Add
your own corpus and QA file with `--corpus` and `--qa` to measure your setup.

## ▸ Development

```bash
pip install -e ".[dev,postgres]"
pytest                      # offline; the pgvector test is skipped without a database
ruff check . && ruff format --check .

docker compose up -d db
NEONRECALL_TEST_PG_DSN=postgresql://neonrecall:neonrecall@localhost:5432/neonrecall pytest -m postgres
```

```
src/neonrecall/
  chunking.py     heading-aware chunker with overlap
  embeddings.py   OpenAI-compatible + hashing embedders
  store/          sqlite.py (also :memory:), pgvector.py
  bm25.py         Okapi BM25
  retrieval.py    hybrid retriever + RRF
  rerank.py       lexical and HTTP rerankers
  generation.py   chat + extractive answerers, citation parsing
  service.py      NeonRecall facade used by every interface
  api.py · cli.py · mcp_server.py · evals.py
```

## ▸ License

MIT. See [LICENSE](LICENSE).
