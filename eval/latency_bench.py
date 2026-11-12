"""/ask latency: cache miss vs cache hit, p50 and p95, measured at the client.

Flushes the answer cache, sends N distinct questions (all misses), then sends
the same N again (all hits). Requests are sequential so each number is
single-request latency, not throughput under load.

  python -m eval.latency_bench [--base http://localhost:8000] [--n 70]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).parent


async def run(base: str, n: int) -> dict:
    qs = [json.loads(l)["question"] for f in ("questions.jsonl", "ooc.jsonl")
          for l in (HERE / f).read_text().splitlines() if l.strip()][:n]
    async with httpx.AsyncClient(base_url=base, timeout=120) as c:
        health = (await c.get("/healthz")).json()
        await c.post("/cache/flush")
        out = {}
        for phase in ("miss", "hit"):
            lat, server, hits = [], [], 0
            for q in qs:
                t = time.perf_counter()
                r = await c.post("/ask", json={"question": q, "use_cache": True})
                lat.append((time.perf_counter() - t) * 1000)
                r.raise_for_status()
                server.append(r.json()["latency_ms"])
                hits += r.headers["x-cache"] == "HIT"
            out[phase] = {"n": len(qs), "cache_hits": hits,
                          "p50_ms": round(float(np.percentile(lat, 50)), 1),
                          "p95_ms": round(float(np.percentile(lat, 95)), 1),
                          "server_p50_ms": round(float(np.percentile(server, 50)), 2),
                          "server_p95_ms": round(float(np.percentile(server, 95)), 2)}
    return {"llm": health["llm"], "model": health["chat_model"], "cache": health["cache"],
            "backend": health["backend"], **out,
            "speedup_p50": round(out["miss"]["p50_ms"] / out["hit"]["p50_ms"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=70)
    a = ap.parse_args()
    res = asyncio.run(run(a.base, a.n))
    (HERE.parent / "bench" / "results" / "latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
