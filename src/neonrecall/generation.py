"""Answer generation with numbered citations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx

from neonrecall.models import Answer, Citation, ScoredChunk
from neonrecall.text import split_sentences, tokenize

NO_ANSWER = "I couldn't find anything in the indexed documents that answers that."

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.
Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
If the passages do not contain the answer, say you don't know. Be concise."""

_CITE_RE = re.compile(r"\[(\d+)\]")


class Answerer(Protocol):
    name: str

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer: ...


def _snippet(text: str, limit: int = 240) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_citations(text: str, contexts: Sequence[ScoredChunk]) -> list[Citation]:
    """Citations for every valid ``[n]`` marker in ``text``, in order of first use."""
    seen: list[int] = []
    for match in _CITE_RE.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(contexts) and n not in seen:
            seen.append(n)
    return [
        Citation(
            index=n,
            chunk_id=contexts[n - 1].chunk.id,
            source=contexts[n - 1].chunk.source,
            title=contexts[n - 1].chunk.title,
            heading=contexts[n - 1].chunk.heading,
            snippet=_snippet(contexts[n - 1].chunk.text),
        )
        for n in seen
    ]


def format_context(contexts: Sequence[ScoredChunk]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        where = " > ".join(p for p in (ctx.chunk.title, ctx.chunk.heading) if p)
        blocks.append(
            f"[{i}] ({ctx.chunk.source}{' — ' + where if where else ''})\n{ctx.chunk.text}"
        )
    return "\n\n".join(blocks)


class ExtractiveAnswerer:
    """Offline fallback: stitches together the most query-relevant sentences from the
    retrieved passages and cites each one."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_relative_score: float = 0.5) -> None:
        self.max_sentences = max_sentences
        self.min_relative_score = min_relative_score

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, int, str]] = []
        if q_terms:
            df: Counter[str] = Counter()
            per_ctx: list[list[tuple[str, set[str]]]] = []
            for ctx in contexts:
                lines = [
                    ln for ln in ctx.chunk.text.splitlines() if not ln.lstrip().startswith("#")
                ]
                sents = [(s, set(tokenize(s))) for s in split_sentences(" ".join(lines))]
                per_ctx.append(sents)
                for _, toks in sents:
                    df.update(toks & q_terms)
            for ci, sents in enumerate(per_ctx):
                for si, (sent, toks) in enumerate(sents):
                    overlap = toks & q_terms
                    if not overlap or len(sent) < 20:
                        continue
                    score = sum(1.0 / df[t] for t in overlap) / len(q_terms)
                    score *= 1.0 / (1.0 + 0.15 * ci)  # trust higher-ranked passages more
                    candidates.append((score, ci, si, sent))
        if not candidates:
            return Answer(question, NO_ANSWER, [], list(contexts), self.name)
        ranked = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))
        cutoff = ranked[0][0] * self.min_relative_score  # drop weak, incidental matches
        best = [c for c in ranked if c[0] >= cutoff][: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))  # keep document order for readability
        text = " ".join(f"{sent} [{ci + 1}]" for _, ci, _, sent in best)
        return Answer(question, text, build_citations(text, contexts), list(contexts), self.name)


class LLMError(RuntimeError):
    pass


class OpenAIChatAnswerer:
    """Calls ``POST {base_url}/chat/completions`` on any OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        if not contexts:
            return Answer(question, NO_ANSWER, [], [], self.name)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context passages:\n\n{format_context(contexts)}\n\n"
                f"Question: {question}",
            },
        ]
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            ValueError,
            AttributeError,
            TypeError,
        ) as exc:
            raise LLMError(f"chat completion failed: {exc}") from exc
        return Answer(question, text, build_citations(text, contexts), list(contexts), self.name)
