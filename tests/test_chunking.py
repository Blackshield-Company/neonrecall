import pytest

from neonrecall.chunking import chunk_document, chunk_text, document_title, split_sections
from neonrecall.models import Document


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        # The next chunk starts with text carried over from the end of the previous one.
        head = nxt.split("\n\n", 1)[0]
        assert prev.endswith(head)


def test_long_sentence_and_token_are_split():
    text = "x" * 1000 + " " + " ".join(["alpha"] * 200)
    chunks = chunk_text(text, size=128, overlap=16)
    assert all(len(c) <= 128 for c in chunks)
    assert "".join(chunks).count("x") >= 1000


def test_no_content_is_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), size=200, overlap=40)
    seen = set(" ".join(chunks).split())
    assert seen == set(words)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)


def test_sections_track_heading_path_and_ignore_code_fences():
    md = "# Top\nintro\n## A\nalpha\n```\n# not a heading\n```\n### A1\ndeep\n## B\nbeta\n"
    sections = split_sections(md)
    assert [s.heading for s in sections] == ["Top", "Top > A", "Top > A > A1", "Top > B"]
    assert "# not a heading" in sections[1].text


def test_chunk_document_ids_and_metadata():
    doc = Document("docs/guide.md", "# Guide\nIntro.\n## Setup\nInstall it.\n")
    chunks = chunk_document(doc, size=200, overlap=20)
    assert [c.ordinal for c in chunks] == [0, 1]
    assert chunks[0].id == f"{doc.doc_id}:0"
    assert chunks[1].heading == "Guide > Setup"
    assert chunks[1].title == "Guide"
    assert chunks[1].embedding_text().startswith("Guide > Guide > Setup\n")


def test_title_falls_back_to_filename():
    assert document_title(Document("notes/night_shift-log.txt", "plain text")) == "night shift log"
    assert document_title(Document("x.md", "# Real Title\nbody", title="Override")) == "Override"
