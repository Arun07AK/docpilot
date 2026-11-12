"""FAISS vs Chroma on identical vectors: build time, query latency p50/p95, disk size.

Embeddings are computed once and reused, so only the vector store is measured.
  python -m bench.store_bench
"""

from __future__ import annotations

import asyncio
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from docpilot.embed.base import get_embedder
from docpilot.ingest.pipeline import chunk_corpus
from docpilot.store.chroma_store import ChromaStore
from docpilot.store.faiss_store import FaissStore

OUT = Path(__file__).parent / "results"
QUERIES = 500


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def pct(xs, q):
    return float(np.percentile(xs, q))


def main():
    chunks, _ = chunk_corpus(256, 32, workers=1)
    emb = get_embedder("local")
    vecs = asyncio.run(emb.embed([c.text for c in chunks]))
    qs = [json.loads(l)["question"] for l in (Path(__file__).parents[1] / "eval" / "questions.jsonl").read_text().splitlines()]
    qv = asyncio.run(emb.embed(qs))
    rows = []
    for name, cls in (("faiss", FaissStore), ("chroma", ChromaStore)):
        tmp = Path(tempfile.mkdtemp())
        try:
            t = time.perf_counter()
            store = cls(tmp)
            store.add(chunks, vecs)
            store.persist()
            build = time.perf_counter() - t
            for v in qv[:10]:  # warm-up
                store.search(v, 5)
            lat = []
            for i in range(QUERIES):
                t = time.perf_counter()
                store.search(qv[i % len(qv)], 5)
                lat.append((time.perf_counter() - t) * 1000)
            rows.append({"backend": name, "vectors": len(chunks), "dim": int(vecs.shape[1]),
                         "build_s": round(build, 3), "query_p50_ms": round(pct(lat, 50), 3),
                         "query_p95_ms": round(pct(lat, 95), 3), "disk_mb": round(du(tmp) / 1e6, 2),
                         "queries": QUERIES})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    (OUT / "stores.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
