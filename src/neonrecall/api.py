"""FastAPI app: POST /ingest, POST /query, GET /health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from neonrecall import __version__
from neonrecall.ingest import load_documents
from neonrecall.models import Document
from neonrecall.service import NeonRecall


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, description="Stable identifier, e.g. a path or URL")
    text: str
    title: str = ""


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(default_factory=list)
    paths: list[str] = Field(
        default_factory=list, description="Files/dirs on the server, below ingest_root"
    )

    @model_validator(mode="after")
    def _non_empty(self) -> IngestRequest:
        if not self.documents and not self.paths:
            raise ValueError("provide at least one of 'documents' or 'paths'")
        return self


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank: bool | None = None
    answer: bool = Field(default=True, description="Generate a cited answer, not just hits")


def _resolve_under(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(400, f"path escapes ingest_root: {raw}")
    if not path.exists():
        raise HTTPException(404, f"no such path: {raw}")
    return path


def create_app(service: NeonRecall | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            app.state.service = NeonRecall.from_settings()
            app.state.owns_service = True
        yield
        if getattr(app.state, "owns_service", False):
            app.state.service.close()

    app = FastAPI(title="Neonrecall", version=__version__, lifespan=lifespan)
    app.state.service = service

    def svc() -> NeonRecall:
        if app.state.service is None:
            app.state.service = NeonRecall.from_settings()
            app.state.owns_service = True
        return app.state.service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, **svc().stats()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        service = svc()
        docs = [Document(d.source, d.text, d.title) for d in req.documents]
        report = service.ingest_documents(docs).to_dict()
        if req.paths:
            root = Path(service.settings.ingest_root).resolve()
            paths = [_resolve_under(root, p) for p in req.paths]
            try:
                more = service.ingest_documents(
                    load_documents(paths, base=root, root=root)
                ).to_dict()
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            report = {
                "documents": report["documents"] + more["documents"],
                "chunks": report["chunks"] + more["chunks"],
                "sources": report["sources"] + more["sources"],
            }
        return report

    @app.post("/query")
    def query(req: QueryRequest) -> dict:
        service = svc()
        if req.answer:
            return service.ask(req.query, req.k, mode=req.mode, rerank=req.rerank).to_dict()
        hits = service.search(req.query, req.k, mode=req.mode, rerank=req.rerank)
        return {"question": req.query, "results": [h.to_dict() for h in hits]}

    return app


def __getattr__(name: str):
    # ``uvicorn neonrecall.api:app`` builds the app lazily from the environment.
    if name == "app":
        return create_app()
    raise AttributeError(name)
