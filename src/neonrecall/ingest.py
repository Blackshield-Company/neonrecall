"""Load Markdown and plain-text documents from disk."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from neonrecall.models import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".mdx", ".txt", ".text", ".rst"})


def iter_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if (
                    child.is_file()
                    and child.suffix.lower() in SUPPORTED_SUFFIXES
                    and not any(part.startswith(".") for part in child.relative_to(path).parts)
                ):
                    yield child
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"unsupported file type: {path}")
            yield path
        else:
            raise FileNotFoundError(path)


def load_documents(paths: Iterable[str | Path], base: str | Path | None = None) -> list[Document]:
    """Read files into Documents. ``source`` is relative to ``base`` when possible."""
    base_path = Path(base).resolve() if base else None
    docs: list[Document] = []
    for file in iter_files(paths):
        resolved = file.resolve()
        source = file.as_posix()
        if base_path is not None and resolved.is_relative_to(base_path):
            source = resolved.relative_to(base_path).as_posix()
        text = file.read_text(encoding="utf-8", errors="replace")
        docs.append(Document(source=source, text=text))
    return docs
