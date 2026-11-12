"""Parsing, chunking and the ingestion pipeline (including the process pool and incremental re-ingest)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docpilot.ingest.chunk import chunk_section, content_hash, encoder
from docpilot.ingest.parse import Section, parse_markdown
from docpilot.ingest.pipeline import chunk_corpus, discover, ingest
from docpilot.store.faiss_store import FaissStore


def _sections(corpus: Path, rel: str, doc: str):
    return parse_markdown(corpus / rel, doc, corpus / "fastapi" / "docs_src")


def test_heading_paths_and_code_comment_is_not_a_heading(mini_corpus):
    secs = _sections(mini_corpus, "fastapi/docs/en/docs/tutorial/path-params.md", "fastapi/tutorial/path-params.md")
    paths = [s.heading_path for s in secs]
    assert paths == ["Path Parameters", "Path Parameters > Path parameters with types", "Path Parameters > Order matters"]
    typed = secs[1].text
    assert "# this comment is not a heading" in typed  # stayed inside the code block


def test_anchor_attributes_and_admonition_markers_are_stripped(mini_corpus):
    secs = _sections(mini_corpus, "fastapi/docs/en/docs/tutorial/path-params.md", "fastapi/tutorial/path-params.md")
    order = secs[2]
    assert "{ #order-matters }" not in order.heading_path
    assert "///" not in order.text and "Tip:" in order.text
    assert "would match also for /users/me" in order.text


def test_page_without_h1_takes_title_from_filename_and_dedents_admonitions(mini_corpus):
    secs = parse_markdown(mini_corpus / "pydantic/docs/concepts/models.md", "pydantic/concepts/models.md")
    assert [s.heading_path for s in secs] == ["Models > Basic model usage", "Models > Extra data"]
    assert "Strictness:" in secs[1].text
    assert "\nValidation in lax mode" in secs[1].text  # 4-space admonition indent removed


def test_code_include_is_inlined(tmp_path):
    src = tmp_path / "docs_src" / "body"
    src.mkdir(parents=True)
    (src / "tutorial001.py").write_text("from pydantic import BaseModel\n")
    md = tmp_path / "body.md"
    md.write_text("# Body\n\n## Model\n\nUse a model:\n\n{* ../../docs_src/body/tutorial001.py hl[1] *}\n")
    (sec,) = [s for s in parse_markdown(md, "fastapi/body.md", tmp_path / "docs_src") if s.heading_path.endswith("Model")]
    assert "```python\nfrom pydantic import BaseModel\n```" in sec.text


def test_chunks_respect_budget_overlap_and_section_boundary():
    words = " ".join(f"word{i}." for i in range(900))
    sec = Section("d.md", "Title", "Title > Long", words)
    chunks = chunk_section(sec, max_tokens=128, overlap=16)
    enc = encoder()
    assert len(chunks) > 3
    for c in chunks:
        assert len(enc.encode(c.text)) <= 128 + 2
        assert c.text.startswith("Title > Long\n\n")  # heading prefix on every chunk
        assert c.heading_path == "Title > Long"
    # overlap: the start of chunk i+1 repeats the end of chunk i
    a = chunks[0].text.split("\n\n", 1)[1]
    b = chunks[1].text.split("\n\n", 1)[1]
    assert a[-20:].strip().split()[-1] in b[:200]


def test_small_paragraphs_are_packed_and_code_blocks_kept_whole():
    text = "Intro para.\n\n```python\nx = 1\n\ny = 2\n```\n\nOutro para."
    (c,) = chunk_section(Section("d.md", "T", "T > S", text), 256, 32)
    assert "```python\nx = 1\n\ny = 2\n```" in c.text


def test_ids_are_content_addressed_and_whitespace_insensitive():
    s1 = Section("d.md", "T", "T > S", "Same   text here.")
    s2 = Section("other.md", "T", "T > S", "Same text here.")
    assert chunk_section(s1, 256, 32)[0].id == chunk_section(s2, 256, 32)[0].id
    assert content_hash("a  b\n c") == content_hash("a b c")


def test_discover_excludes_reference_and_private_pages(mini_corpus):
    ref = mini_corpus / "fastapi/docs/en/docs/reference/apirouter.md"
    ref.parent.mkdir(parents=True)
    ref.write_text("# APIRouter\n\n::: fastapi.APIRouter\n")
    (mini_corpus / "fastapi/docs/en/docs/_llm-test.md").write_text("# x\n\n## y\n\nz\n")
    docs = [d for _, d, _ in discover(mini_corpus)]
    assert "fastapi/reference/apirouter.md" not in docs and "fastapi/_llm-test.md" not in docs
    assert {"fastapi/tutorial/path-params.md", "fastapi/tutorial/cors.md", "pydantic/concepts/models.md"} <= set(docs)


def test_process_pool_gives_identical_chunks_to_serial(mini_corpus):
    serial, st1 = chunk_corpus(64, 8, workers=1, corpus_dir=mini_corpus)
    pooled, st2 = chunk_corpus(64, 8, workers=2, corpus_dir=mini_corpus)
    assert [c.id for c in serial] == [c.id for c in pooled]
    assert st1.sections == st2.sections == 6


async def test_incremental_ingest_skips_unchanged_and_removes_stale(mini_corpus, tmp_path, fake_embedder):
    store = FaissStore(tmp_path / "idx")
    first = await ingest(store, fake_embedder, 64, 8, workers=1, corpus_dir=mini_corpus)
    assert first.new == first.chunks > 0 and first.skipped == 0
    again = await ingest(store, fake_embedder, 64, 8, workers=1, corpus_dir=mini_corpus)
    assert again.new == 0 and again.skipped == first.chunks and again.removed == 0

    (mini_corpus / "fastapi/docs/en/docs/tutorial/cors.md").write_text(
        "# CORS\n\n## Use CORSMiddleware\n\nNew wording: add CORSMiddleware with allow_origins.\n")
    changed = await ingest(store, fake_embedder, 64, 8, workers=1, corpus_dir=mini_corpus)
    assert changed.new == 1 and changed.removed == 1
    assert store.count() == first.chunks
    reloaded = FaissStore(tmp_path / "idx")
    assert reloaded.ids() == store.ids()
