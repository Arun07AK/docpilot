"""Command line: `python -m docpilot.cli <command>`.

  fetch                         clone the pinned corpus
  chunk-stats                   chunk counts + serial vs pool timing
  ingest  --backend --chunk     build one index (incremental)
  ingest-all                    build every index the retrieval eval needs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from docpilot.config import settings

CHUNK_CONFIGS = {256: 32, 512: 64}


def cmd_fetch(_):
    from docpilot.ingest.fetch import fetch

    for name, root in fetch().items():
        print(name, root)


def cmd_chunk_stats(args):
    from docpilot.ingest.pipeline import chunk_corpus

    out = {}
    for size, ov in CHUNK_CONFIGS.items():
        t = time.perf_counter()
        chunks, st = chunk_corpus(size, ov, workers=1)
        serial = time.perf_counter() - t
        t = time.perf_counter()
        chunks2, _ = chunk_corpus(size, ov, workers=args.workers or None)
        pool = time.perf_counter() - t
        assert [c.id for c in chunks] == [c.id for c in chunks2], "pool output differs"
        out[size] = {"files": st.files, "sections": st.sections, "chunks": st.chunks,
                     "tokens": sum(c.token_count for c in chunks),
                     "serial_s": round(serial, 3), "pool_s": round(pool, 3)}
        print(size, out[size])
    (settings.data_dir.parent / "bench" / "results" / "chunk_stats.json").write_text(json.dumps(out, indent=2))


async def _ingest(backend: str, size: int, provider: str, workers: int | None):
    from docpilot.embed.base import get_embedder
    from docpilot.ingest.pipeline import ingest
    from docpilot.store.base import get_store

    store = get_store(backend, settings.index_dir(backend, size, provider))
    emb = get_embedder(provider)
    t = time.perf_counter()
    st = await ingest(store, emb, size, CHUNK_CONFIGS.get(size, size // 8), workers=workers)
    print(f"{backend}/{provider}/{size}: files={st.files} chunks={st.chunks} new={st.new} "
          f"skipped={st.skipped} removed={st.removed} total={time.perf_counter() - t:.1f}s "
          f"timings={ {k: round(v, 2) for k, v in st.timings.items()} }")
    return st


def cmd_ingest(args):
    asyncio.run(_ingest(args.backend, args.chunk, args.provider, args.workers))


def cmd_ingest_all(args):
    for size in CHUNK_CONFIGS:
        for backend in ("faiss", "chroma"):
            asyncio.run(_ingest(backend, size, args.provider, args.workers))


def main():
    p = argparse.ArgumentParser(prog="docpilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    s = sub.add_parser("chunk-stats")
    s.add_argument("--workers", type=int, default=0)
    s.set_defaults(fn=cmd_chunk_stats)
    s = sub.add_parser("ingest")
    s.add_argument("--backend", default=settings.vector_backend, choices=["faiss", "chroma"])
    s.add_argument("--chunk", type=int, default=settings.chunk_tokens)
    s.add_argument("--provider", default=settings.embed_provider, choices=["local", "openai"])
    s.add_argument("--workers", type=int, default=None)
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("ingest-all")
    s.add_argument("--provider", default=settings.embed_provider, choices=["local", "openai"])
    s.add_argument("--workers", type=int, default=None)
    s.set_defaults(fn=cmd_ingest_all)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
