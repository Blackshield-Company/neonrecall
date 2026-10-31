"""Neonrecall: hybrid-retrieval RAG with citations."""

from neonrecall.models import Answer, Chunk, Citation, Document, ScoredChunk
from neonrecall.service import NeonRecall

__all__ = ["Answer", "Chunk", "Citation", "Document", "NeonRecall", "ScoredChunk"]
__version__ = "0.1.0"
