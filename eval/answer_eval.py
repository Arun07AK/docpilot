"""Answer behaviour through the live API (/ask, cache off).

  in-corpus (50): answer rate, and the share of answers that cite the gold section
  out-of-corpus (20): refusal rate (correctly said "insufficient evidence")

Run with the server up and OPENAI_API_KEY set:  python -m eval.answer_eval [--base http://localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def load(name):
    return [json.loads(l) for l in (HERE / name).read_text().splitlines() if l.strip()]


async def run(base: str, concurrency: int) -> dict:
    qs, ooc = load("questions.jsonl"), load("ooc.jsonl")
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base, timeout=120) as c:
        health = (await c.get("/healthz")).json()
        assert health["llm"], "answer eval needs the LLM path (set OPENAI_API_KEY and restart the server)"

        async def ask(q):
            async with sem:
                r = await c.post("/ask", json={"question": q["question"], "use_cache": False})
                r.raise_for_status()
                return q, r.json()

        res_in = await asyncio.gather(*(ask(q) for q in qs))
        res_out = await asyncio.gather(*(ask(q) for q in ooc))

    answered = [(q, r) for q, r in res_in if not r["refused"]]
    cited_gold = [q for q, r in answered if any(c["doc_path"] == q["gold_doc"] and c["heading_path"] == q["gold_heading"]
                                               for c in r["citations"])]
    cited_doc = [q for q, r in answered if any(c["doc_path"] == q["gold_doc"] for c in r["citations"])]
    refused_ooc = [q for q, r in res_out if r["refused"]]
    by_kind = {}
    for q, r in res_out:
        k = by_kind.setdefault(q["kind"], [0, 0])
        k[0] += r["refused"]
        k[1] += 1
    tok = sum(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for _, r in res_in + res_out)
    summary = {
        "model": health["chat_model"], "backend": health["backend"], "chunk_tokens": health["chunk_tokens"],
        "in_corpus": len(qs), "answered": len(answered), "answer_rate": round(len(answered) / len(qs), 3),
        "false_refusal_rate": round(1 - len(answered) / len(qs), 3),
        "cited_gold_section_rate": round(len(cited_gold) / max(1, len(answered)), 3),
        "cited_gold_doc_rate": round(len(cited_doc) / max(1, len(answered)), 3),
        "out_of_corpus": len(ooc), "ooc_refused": len(refused_ooc),
        "ooc_refusal_rate": round(len(refused_ooc) / len(ooc), 3),
        "ooc_refusal_by_kind": {k: f"{a}/{b}" for k, (a, b) in by_kind.items()},
        "rewrites_used": sum(r["rewrites"] for _, r in res_in + res_out),
        "total_tokens": tok,
        "failures": {"false_refusals": [q["id"] for q, r in res_in if r["refused"]],
                     "ooc_answered": [{"id": q["id"], "q": q["question"], "answer": r["answer"][:200]}
                                      for q, r in res_out if not r["refused"]]},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    s = asyncio.run(run(a.base, a.concurrency))
    (HERE.parent / "bench" / "results" / "answers.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k != "failures"}, indent=2))
    print("failures:", json.dumps(s["failures"], indent=2))


if __name__ == "__main__":
    main()
