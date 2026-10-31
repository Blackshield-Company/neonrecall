import asyncio

from neonrecall.mcp_server import build_server


def test_mcp_lists_tools(loaded):
    server = build_server(loaded)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"search_docs", "ask"}


def test_mcp_search_and_ask(loaded):
    server = build_server(loaded)
    res = asyncio.run(server.call_tool("search_docs", {"query": "gated reverb snare", "k": 2}))
    assert not res.is_error
    hits = res.structured_content["result"]
    assert len(hits) == 2 and hits[0]["source"] == "drum-machine-programming.md"

    res = asyncio.run(server.call_tool("ask", {"question": "Which gas glows blue?"}))
    body = res.structured_content
    assert "Argon" in body["answer"]
    assert body["citations"][0]["source"] == "neon-sign-care.md"
