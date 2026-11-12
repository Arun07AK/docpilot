"""The /ask agent as a LangGraph state machine.

    retrieve -> grade --sufficient--------------> answer --citations ok--> END
                  |                                  '---no/invalid cites--> refuse
                  |--insufficient, rewrites < 1----> rewrite -> retrieve
                  '--insufficient, already rewrote-> refuse -> END

Why a graph rather than one prompt: each step is small, testable and logged on
its own, and the control flow (one rewrite, then refuse) is explicit code
instead of something a prompt hopes the model will follow. Refusal is the
default whenever evidence is weak: the grader must say "sufficient", and every
citation in the answer must point at a passage the grader marked relevant.

With no OpenAI key, the grader falls back to a similarity threshold and the
answer step returns the best passages verbatim (mode="extractive"), so the
service still runs offline.
"""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from docpilot.agent import llm
from docpilot.agent.prompts import (ANSWER_SCHEMA, ANSWER_SYSTEM, GRADER_SCHEMA, GRADER_SYSTEM,
                                    REFUSAL, REWRITE_SYSTEM)
from docpilot.config import settings

log = structlog.get_logger()

MAX_REWRITES = 1
# Offline fallback threshold on cosine similarity (all-MiniLM-L6-v2). Calibrated on the eval set.
SCORE_THRESHOLD = 0.45


class AskState(TypedDict, total=False):
    question: str
    query: str
    hits: list[dict]
    relevant: list[int]  # 1-based passage numbers the grader kept
    sufficient: bool
    missing: str
    rewrites: int
    answer: str
    citations: list[dict]
    nearest: list[dict]
    refused: bool
    mode: str  # llm | extractive
    prompt_tokens: int
    completion_tokens: int
    trace: list[dict]


def _passages(hits: list[dict], only: list[int] | None = None) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        if only is not None and i not in only:
            continue
        parts.append(f"[{i}] ({h['doc_path']} > {h['heading_path']})\n{h['text']}")
    return "\n\n".join(parts)


def _step(state: AskState, name: str, t0: float, **info) -> list[dict]:
    return [*state.get("trace", []), {"step": name, "ms": round((time.perf_counter() - t0) * 1000, 1), **info}]


def _usage(state: AskState, u: llm.Usage) -> dict:
    return {"prompt_tokens": state.get("prompt_tokens", 0) + u.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + u.completion_tokens}


def build_graph(retriever):
    """`retriever(query) -> list[dict]` is async; injected so tests can fake it."""

    async def retrieve(state: AskState) -> dict:
        t0 = time.perf_counter()
        query = state.get("query") or state["question"]
        hits = await retriever(query)
        return {"query": query, "hits": hits,
                "trace": _step(state, "retrieve", t0, query=query, k=len(hits),
                               top=[{"doc": h["doc_path"], "heading": h["heading_path"],
                                     "score": round(h["score"], 3)} for h in hits])}

    async def grade(state: AskState) -> dict:
        t0 = time.perf_counter()
        hits = state["hits"]
        if not hits:
            return {"relevant": [], "sufficient": False, "missing": "no passages",
                    "trace": _step(state, "grade", t0, relevant=[], sufficient=False)}
        if not llm.available():
            rel = [i for i, h in enumerate(hits, 1) if h["score"] >= SCORE_THRESHOLD]
            return {"relevant": rel, "sufficient": bool(rel), "missing": "",
                    "trace": _step(state, "grade", t0, relevant=rel, sufficient=bool(rel), mode="threshold")}
        msgs = [{"role": "system", "content": GRADER_SYSTEM},
                {"role": "user", "content": f"Question: {state['question']}\n\nPassages:\n{_passages(hits)}"}]
        try:
            out, u = await llm.complete_json(msgs, GRADER_SCHEMA, "grade", max_tokens=200)
        except Exception as exc:
            log.warning("agent.grade_failed", error=str(exc)[:200])
            return {"relevant": [], "sufficient": False, "missing": "grader error",
                    "trace": _step(state, "grade", t0, error=True)}
        rel = sorted({i for i in out["relevant"] if 1 <= i <= len(hits)})
        sufficient = bool(out["sufficient"] and rel)
        return {"relevant": rel, "sufficient": sufficient, "missing": out.get("missing", ""), **_usage(state, u),
                "trace": _step(state, "grade", t0, relevant=rel, sufficient=sufficient)}

    def route(state: AskState) -> str:
        if state.get("sufficient"):
            return "answer"
        if llm.available() and state.get("rewrites", 0) < MAX_REWRITES:
            return "rewrite"
        return "refuse"

    async def rewrite(state: AskState) -> dict:
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"Question: {state['question']}\n"
                                            f"Previous search query: {state.get('query')}\n"
                                            f"What was missing: {state.get('missing') or 'unknown'}"}]
        try:
            q, u = await llm.complete(msgs, max_tokens=60)
            q = q.strip().strip('"') or state["question"]
        except Exception:
            q, u = state["question"], llm.Usage()
        return {"query": q, "rewrites": state.get("rewrites", 0) + 1, **_usage(state, u),
                "trace": _step(state, "rewrite", t0, query=q)}

    async def answer(state: AskState) -> dict:
        t0 = time.perf_counter()
        hits, rel = state["hits"], state["relevant"]
        if not llm.available():
            keep = [hits[i - 1] for i in rel[:3]]
            text = "\n\n".join(f"{h['text'].split(chr(10), 2)[-1].strip()[:600]} [{n}]" for n, h in enumerate(keep, 1))
            return {"answer": text, "citations": _cites(keep), "refused": False, "mode": "extractive",
                    "trace": _step(state, "answer", t0, mode="extractive")}
        msgs = [{"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {state['question']}\n\nPassages:\n{_passages(hits, rel)}"}]
        try:
            out, u = await llm.complete_json(msgs, ANSWER_SCHEMA, "answer", max_tokens=600)
        except Exception as exc:
            log.warning("agent.answer_failed", error=str(exc)[:200])
            return {"refused": True, "trace": _step(state, "answer", t0, error=True)}
        cited_in_text = {int(n) for n in re.findall(r"\[(\d+)\]", out["answer"])}
        cited = cited_in_text | set(out["citations"])
        valid = bool(out["sufficient"] and out["answer"].strip() and cited and cited <= set(rel))
        if not valid:
            return {"refused": True, **_usage(state, u),
                    "trace": _step(state, "answer", t0, rejected=True, cited=sorted(cited), allowed=rel)}
        # renumber citations 1..n in order of first use so the UI shows [1][2] not [4][7]
        order = []
        for n in [int(x) for x in re.findall(r"\[(\d+)\]", out["answer"])] + sorted(cited):
            if n not in order:
                order.append(n)
        remap = {old: new for new, old in enumerate(order, 1)}
        text = re.sub(r"\[(\d+)\]", lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", out["answer"])
        return {"answer": text, "citations": _cites([hits[i - 1] for i in order]), "refused": False,
                "mode": "llm", **_usage(state, u), "trace": _step(state, "answer", t0, cited=order)}

    async def refuse(state: AskState) -> dict:
        t0 = time.perf_counter()
        near = state.get("hits", [])[:3]
        return {"answer": REFUSAL, "refused": True, "citations": [],
                "nearest": [{"doc_path": h["doc_path"], "heading_path": h["heading_path"],
                             "score": round(h["score"], 3)} for h in near],
                "trace": _step(state, "refuse", t0)}

    def after_answer(state: AskState) -> str:
        return "refuse" if state.get("refused") else END

    g = StateGraph(AskState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("answer", answer)
    g.add_node("refuse", refuse)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("answer", after_answer, {"refuse": "refuse", END: END})
    g.add_edge("refuse", END)
    return g.compile()


def _cites(hits: list[dict]) -> list[dict]:
    return [{"n": i, "id": h["id"], "doc_path": h["doc_path"], "heading_path": h["heading_path"],
             "url": doc_url(h["doc_path"], h["heading_path"]), "score": round(h["score"], 3),
             "snippet": h["text"].split("\n", 2)[-1].strip()[:400]} for i, h in enumerate(hits, 1)]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def doc_url(doc_path: str, heading_path: str) -> str:
    """Public docs URL for a citation (best effort: page + last heading anchor)."""
    source, rel = doc_path.split("/", 1)
    page = rel[:-3]
    if page.endswith("index"):
        page = page[: -len("index")]
    base = "https://fastapi.tiangolo.com/" if source == "fastapi" else "https://docs.pydantic.dev/latest/"
    anchor = _slug(heading_path.split(" > ")[-1]) if " > " in heading_path else ""
    return f"{base}{page.strip('/')}/" + (f"#{anchor}" if anchor else "")


async def run_ask(graph, question: str) -> dict[str, Any]:
    state = await graph.ainvoke({"question": question, "rewrites": 0, "trace": [],
                                 "prompt_tokens": 0, "completion_tokens": 0})
    return state
