"""Profile a full cold ingest with cProfile and time it.

  python -m bench.profile_ingest --label before   # writes bench/results/profile_before.{prof,txt}
  python -m bench.profile_ingest --label after --sort-by-length

The cold ingest (fresh index dir, empty embedding cache) is profiled once, then
timed 3 times; the median goes to bench/results/profile.json.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import pstats
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from docpilot.config import settings
from docpilot.embed.base import get_embedder
from docpilot.ingest.pipeline import ingest
from docpilot.store.faiss_store import FaissStore

OUT = Path(__file__).parent / "results"


async def cold_ingest(sort_by_length: bool, chunk_tokens: int, batch_size: int) -> dict:
    tmp = Path(tempfile.mkdtemp())
    try:
        emb = get_embedder("local")
        emb.batch_size = batch_size
        emb.sort_by_length = sort_by_length
        t = time.perf_counter()
        st = await ingest(FaissStore(tmp), emb, chunk_tokens, chunk_tokens // 8, workers=1)
        return {"total_s": time.perf_counter() - t, **{k: v for k, v in st.timings.items()}, "chunks": st.chunks}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--sort-by-length", action="store_true")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batch", type=int, default=settings.embed_batch_size)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    get_embedder("local").warm()  # model load is a one-time cost; keep it out of the measurement
    prof = cProfile.Profile()
    prof.enable()
    asyncio.run(cold_ingest(a.sort_by_length, a.chunk, a.batch))
    prof.disable()
    prof.dump_stats(OUT / f"profile_{a.label}.prof")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
    (OUT / f"profile_{a.label}.txt").write_text(s.getvalue())

    runs = [asyncio.run(cold_ingest(a.sort_by_length, a.chunk, a.batch)) for _ in range(a.runs)]
    med = {k: round(statistics.median(r[k] for r in runs), 3) for k in runs[0] if k != "chunks"}
    path = OUT / "profile.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[a.label] = {"sort_by_length": a.sort_by_length, "batch_size": a.batch, "chunk_tokens": a.chunk,
                     "chunks": runs[0]["chunks"], "runs": a.runs, "median": med,
                     "all_total_s": [round(r["total_s"], 3) for r in runs]}
    path.write_text(json.dumps(data, indent=2))
    print(a.label, json.dumps(data[a.label]))


if __name__ == "__main__":
    main()
