"""Markdown-aware chunking with character overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from neonrecall.models import Chunk, Document
from neonrecall.text import split_sentences

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    heading: str  # "Parent > Child" breadcrumb, empty for preamble
    text: str


def split_sections(markdown: str) -> list[Section]:
    """Split Markdown into sections at headings, ignoring '#' inside fenced code."""
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(" > ".join(t for _, t in path), body))
        buf.clear()

    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
        buf.append(line)
    flush()
    return sections


def _units(text: str, size: int) -> list[str]:
    """Break text into paragraph/sentence/word units no longer than ``size``."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sentence in split_sentences(para):
            if len(sentence) <= size:
                units.append(sentence)
                continue
            words, cur = sentence.split(), ""
            for word in words:
                while len(word) > size:  # pathological token: hard split
                    if cur:
                        units.append(cur)
                        cur = ""
                    units.append(word[:size])
                    word = word[size:]
                if cur and len(cur) + 1 + len(word) > size:
                    units.append(cur)
                    cur = word
                else:
                    cur = f"{cur} {word}" if cur else word
            if cur:
                units.append(cur)
    return units


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    # Start the overlap on a word boundary when one is available.
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Pack text into chunks of at most ``size`` chars, each sharing ~``overlap`` chars
    with its predecessor."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks: list[str] = []
    cur = ""
    for unit in _units(text, size):
        if cur and len(cur) + 2 + len(unit) > size:
            chunks.append(cur)
            cur = _tail(cur, overlap)
            # Drop the carried-over tail if it cannot fit alongside the next unit.
            if len(cur) + 2 + len(unit) > size:
                cur = ""
        cur = f"{cur}\n\n{unit}" if cur else unit
    if cur:
        chunks.append(cur)
    return chunks


def document_title(doc: Document) -> str:
    if doc.title:
        return doc.title
    for line in doc.text.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    stem = doc.source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip() or doc.source


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    title = document_title(doc)
    chunks: list[Chunk] = []
    for section in split_sections(doc.text):
        for piece in chunk_text(section.text, size=size, overlap=overlap):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc.doc_id}:{ordinal}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=title,
                    ordinal=ordinal,
                    text=piece,
                    metadata={"heading": section.heading},
                )
            )
    return chunks
