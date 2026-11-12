"""Ingestion pipeline: discover, then parse+chunk (process pool), diff hashes, embed (async), index.

Why two concurrency models:
  * parse + tokenise is pure-Python CPU work. Threads would serialise on the GIL,
    so a multiprocessing.Pool spreads files across cores;
  * embedding is I/O (OpenAI) or a native batched op (torch), so it runs on the
    event loop with batching and a semaphore.

Re-ingesting is incremental. Chunk ids are content hashes, so an unchanged chunk
has an id the store already holds and is skipped. Chunks whose doc changed
or disappeared are deleted.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

from docpilot.config import settings
from docpilot.ingest.chunk import Chunk, chunk_section
from docpilot.ingest.parse import parse_markdown

# Pages that are navigation, people lists, changelogs or translations rather than docs.
_EXCLUDE_PARTS = {
    "release-notes.md", "fastapi-people.md", "external-links.md", "newsletter.md",
    "contributing.md", "management.md", "management-tasks.md", "help-fastapi.md",
    "translation-banner.md", "sponsors.md", "changelog.md", "llms.txt",
}
_EXCLUDE_DIRS = {"api", "plugins", "theme", "overrides", "img", "badges", "extra", "integrations/llms"}


@dataclass
class FileTask:
    path: Path
    doc_path: str
    docs_src: Path | None
    chunk_tokens: int
    overlap: int


@dataclass
class IngestStats:
    files: int = 0
    sections: int = 0
    chunks: int = 0
    new: int = 0
    skipped: int = 0
    removed: int = 0
    timings: dict[str, float] = field(default_factory=dict)


def discover(corpus_dir: Path | None = None) -> list[tuple[Path, str, Path | None]]:
    corpus_dir = corpus_dir or settings.corpus_dir
    found = []
    layout = {"fastapi": ("docs/en/docs", "docs_src"), "pydantic": ("docs", None)}
    for name, (docs_rel, src_rel) in layout.items():
        root = corpus_dir / name / docs_rel
        if not root.exists():
            continue
        src = corpus_dir / name / src_rel if src_rel else None
        for p in sorted(root.rglob("*.md")):
            rel = p.relative_to(root)
            if (p.name in _EXCLUDE_PARTS or p.name.startswith("_")
                    or set(rel.parts[:-1]) & _EXCLUDE_DIRS
                    or (name == "fastapi" and rel.parts[0] == "reference")):
                continue
            found.append((p, f"{name}/{rel.as_posix()}", src))
    return found


def process_file(task: FileTask) -> tuple[int, list[Chunk]]:
    """Runs inside a pool worker: parse one file, chunk every section."""
    sections = parse_markdown(task.path, task.doc_path, task.docs_src)
    chunks: list[Chunk] = []
    for sec in sections:
        chunks.extend(chunk_section(sec, task.chunk_tokens, task.overlap))
    return len(sections), chunks


def chunk_corpus(chunk_tokens: int, overlap: int, workers: int | None = None,
                 corpus_dir: Path | None = None) -> tuple[list[Chunk], IngestStats]:
    files = discover(corpus_dir)
    tasks = [FileTask(p, d, s, chunk_tokens, overlap) for p, d, s in files]
    workers = workers if workers is not None else (settings.ingest_workers or os.cpu_count() or 1)
    stats = IngestStats(files=len(tasks))
    t0 = time.perf_counter()
    if workers <= 1:
        results = [process_file(t) for t in tasks]
    else:
        # spawn: safe with torch/faiss already imported in the parent
        with get_context("spawn").Pool(workers) as pool:
            results = pool.map(process_file, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
    stats.timings["parse_chunk_s"] = time.perf_counter() - t0

    seen: set[str] = set()
    chunks: list[Chunk] = []
    for n_sec, cs in results:
        stats.sections += n_sec
        for c in cs:
            if c.id in seen:  # identical text in two places: index once
                continue
            seen.add(c.id)
            chunks.append(c)
    stats.chunks = len(chunks)
    return chunks, stats


async def ingest(store, embedder, chunk_tokens: int | None = None, overlap: int | None = None,
                 workers: int | None = None, corpus_dir: Path | None = None,
                 on_progress: Callable[[str, dict], None] | None = None,
                 on_chunks: Callable[[list[Chunk]], object] | None = None) -> IngestStats:
    """Full incremental ingest into `store`. Returns counts and per-stage timings."""
    import asyncio

    chunk_tokens = chunk_tokens or settings.chunk_tokens
    overlap = settings.chunk_overlap if overlap is None else overlap
    report = on_progress or (lambda stage, info: None)

    report("parsing", {})
    chunks, stats = await asyncio.to_thread(chunk_corpus, chunk_tokens, overlap, workers, corpus_dir)
    report("parsed", {"files": stats.files, "chunks": stats.chunks})

    existing = store.ids()
    wanted = {c.id for c in chunks}
    new = [c for c in chunks if c.id not in existing]
    stale = existing - wanted
    stats.new, stats.skipped, stats.removed = len(new), len(chunks) - len(new), len(stale)
    if stale:
        store.delete(list(stale))

    t0 = time.perf_counter()
    if new:
        vectors = await embedder.embed([c.text for c in new],
                                       on_batch=lambda done, total: report("embedding", {"done": done, "total": total}))
        stats.timings["embed_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        store.add(new, vectors)
        store.persist()
        stats.timings["index_s"] = time.perf_counter() - t1
    if on_chunks is not None:
        res = on_chunks(chunks)
        if asyncio.iscoroutine(res):
            await res
    report("done", {"new": stats.new, "skipped": stats.skipped, "removed": stats.removed})
    return stats
