import json

import httpx
import pytest

from neonrecall.generation import (
    NO_ANSWER,
    ExtractiveAnswerer,
    LLMError,
    OpenAIChatAnswerer,
    build_citations,
)
from neonrecall.models import Chunk, ScoredChunk


def _ctx(*texts):
    return [
        ScoredChunk(
            Chunk(f"d{i}:0", f"d{i}", f"doc{i}.md", f"Doc {i}", 0, t, {"heading": "H"}), 1.0
        )
        for i, t in enumerate(texts)
    ]


def test_build_citations_dedupes_and_ignores_out_of_range():
    ctx = _ctx("alpha text", "beta text")
    cites = build_citations("Foo [2]. Bar [1][2]. Baz [7].", ctx)
    assert [c.index for c in cites] == [2, 1]
    assert cites[0].source == "doc1.md"
    assert cites[0].heading == "H"


def test_extractive_answer_cites_sources():
    ctx = _ctx(
        "# Heading\nStore tape at 15 to 20 degrees. Keep humidity between 30 and 40 percent.",
        "Neon glows red. Argon glows blue.",
    )
    ans = ExtractiveAnswerer().answer("What humidity should tape be stored at?", ctx)
    assert "humidity" in ans.text and "[1]" in ans.text
    assert "Heading" not in ans.text
    assert [c.source for c in ans.citations] == ["doc0.md"]
    assert ans.backend == "extractive"


def test_extractive_no_answer():
    ans = ExtractiveAnswerer().answer("quantum chromodynamics", _ctx("Neon glows red."))
    assert ans.text == NO_ANSWER and ans.citations == []
    assert ExtractiveAnswerer().answer("anything", []).text == NO_ANSWER


def test_openai_chat_answerer_sends_context_and_parses_citations():
    sent = {}

    def handler(request):
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Argon [2]."}}]})

    llm = OpenAIChatAnswerer(
        "http://ollama:11434/v1",
        "llama3.1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ans = llm.answer("blue gas?", _ctx("Neon glows red.", "Argon glows blue."))
    assert sent["path"] == "/v1/chat/completions"
    user_msg = sent["body"]["messages"][1]["content"]
    assert "[2] (doc1.md — Doc 1 > H)" in user_msg and "blue gas?" in user_msg
    assert ans.text == "Argon [2]."
    assert [c.index for c in ans.citations] == [2]


def test_openai_chat_answerer_errors():
    llm = OpenAIChatAnswerer(
        "http://x/v1",
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))
        ),
    )
    with pytest.raises(LLMError):
        llm.answer("q", _ctx("text"))
    assert llm.answer("q", []).text == NO_ANSWER
