"""MCP server exposing ``search_docs`` and ``ask`` over stdio (or streamable HTTP)."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from neonrecall.service import NeonRecall

INSTRUCTIONS = (
    "Neonrecall searches a private document index. Use search_docs to find passages and "
    "ask to get a short answer with numbered citations to source documents."
)


def build_server(service: NeonRecall) -> MCPServer:
    server = MCPServer("neonrecall", instructions=INSTRUCTIONS)

    @server.tool()
    def search_docs(
        query: str, k: int = 5, mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    ) -> list[dict[str, Any]]:
        """Search the indexed documents. Returns the top-k passages with source, heading
        and relevance score."""
        k = max(1, min(k, 50))
        return [
            {
                "chunk_id": h.chunk.id,
                "source": h.chunk.source,
                "title": h.chunk.title,
                "heading": h.chunk.heading,
                "score": round(h.score, 6),
                "text": h.chunk.text,
            }
            for h in service.search(query, k, mode=mode)
        ]

    @server.tool()
    def ask(question: str, k: int = 5) -> dict[str, Any]:
        """Answer a question from the indexed documents. The answer cites passages as
        [n]; each citation lists its source document."""
        answer = service.ask(question, max(1, min(k, 50)))
        return {
            "answer": answer.text,
            "citations": [c.to_dict() for c in answer.citations],
            "backend": answer.backend,
        }

    return server


def main(transport: Literal["stdio", "streamable-http"] = "stdio") -> None:
    service = NeonRecall.from_settings()
    try:
        build_server(service).run(transport)
    finally:
        service.close()


if __name__ == "__main__":
    main()
