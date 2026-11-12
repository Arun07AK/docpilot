"""Retrieval eval: recall@k and MRR at the section level.

A retrieved chunk counts as a hit when its (doc_path, heading_path) equals the
question's gold section. Scoring at section level (not chunk id) is what makes
256-token and 512-token chunkings comparable: both index the same sections.

  recall@k = share of questions with >=1 gold-section chunk in the top k
  MRR      = mean over questions of 1/rank of the first gold-section chunk (0 if none in top k)

Run:  python -m eval.retrieval_eval [--k 5] [--providers local,openai]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from docpilot.config import settings
from docpilot.embed.base import get_embedder
from docpilot.store.base import get_store

HERE = Path(__file__).parent


def load_questions() -> list[dict]:
    return [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]


def score(ranked: list[list[tuple[str, str]]], gold: list[tuple[str, str]], k: int) -> dict:
    hits, rr = 0, 0.0
    for rk, g in zip(ranked, gold):
        for i, sec in enumerate(rk[:k]):
            if sec == g:
                hits += 1
                rr += 1.0 / (i + 1)
                break
    n = len(gold)
    return {"recall@k": hits / n, "mrr": rr / n, "n": n}


async def run(k: int, providers: list[str]) -> list[dict]:
    qs = load_questions()
    gold = [(q["gold_doc"], q["gold_heading"]) for q in qs]
    rows = []
    for provider in providers:
        emb = get_embedder(provider)
        qv = await emb.embed([q["question"] for q in qs])
        for size in (256, 512):
            for backend in ("faiss", "chroma"):
                path = settings.index_dir(backend, size, provider)
                if not path.exists():
                    continue
                store = get_store(backend, path)
                t = time.perf_counter()
                ranked = [[(h.meta["doc_path"], h.meta["heading_path"]) for h in store.search(v, k)] for v in qv]
                ms = (time.perf_counter() - t) * 1000 / len(qs)
                r = score(ranked, gold, k)
                rows.append({"backend": backend, "provider": provider, "chunk_tokens": size,
                             "k": k, "recall": round(r["recall@k"], 3), "mrr": round(r["mrr"], 3),
                             "n": r["n"], "chunks": store.count(), "search_ms": round(ms, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--providers", default="local")
    a = ap.parse_args()
    rows = asyncio.run(run(a.k, a.providers.split(",")))
    out = HERE.parent / "bench" / "results" / "retrieval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"{'backend':8} {'embed':7} {'chunk':>5} {'recall@'+str(a.k):>9} {'MRR':>6} {'chunks':>6} {'ms/q':>6}")
    for r in rows:
        print(f"{r['backend']:8} {r['provider']:7} {r['chunk_tokens']:>5} {r['recall']:>9.3f} {r['mrr']:>6.3f} {r['chunks']:>6} {r['search_ms']:>6.2f}")


if __name__ == "__main__":
    main()
