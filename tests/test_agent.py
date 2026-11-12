"""Every branch of the LangGraph agent, with the LLM seam scripted (no network)."""

from __future__ import annotations

import json

import pytest

from docpilot.agent import graph as G
from docpilot.agent import llm
from docpilot.agent.prompts import REFUSAL


def hit(i, score=0.8, heading="Path Parameters > Types"):
    return {"id": f"id{i}", "score": score, "text": f"{heading}\n\npassage {i} about path params",
            "doc_path": "fastapi/tutorial/path-params.md", "heading_path": heading, "title": "Path Parameters"}


class Script:
    """Replays queued LLM outputs and records which step asked."""

    def __init__(self, monkeypatch, outputs):
        self.outputs = list(outputs)
        self.seen = []
        monkeypatch.setattr(llm, "available", lambda: True)

        async def complete_json(messages, schema, name="output", max_tokens=700):
            self.seen.append(name)
            return self.outputs.pop(0), llm.Usage(100, 20)

        async def complete(messages, schema=None, name="output", temperature=0.0, max_tokens=700):
            self.seen.append("rewrite")
            return self.outputs.pop(0), llm.Usage(50, 10)

        monkeypatch.setattr(llm, "complete_json", complete_json)
        monkeypatch.setattr(llm, "complete", complete)


def retriever_returning(*batches):
    calls = []

    async def r(query):
        calls.append(query)
        return batches[min(len(calls) - 1, len(batches) - 1)]

    r.calls = calls
    return r


async def run(retriever, q="How do I type a path parameter?"):
    return await G.run_ask(G.build_graph(retriever), q)


async def test_answer_with_valid_citations(monkeypatch):
    s = Script(monkeypatch, [
        {"relevant": [1, 2], "sufficient": True, "missing": ""},
        {"sufficient": True, "answer": "Annotate it with a type [2]. FastAPI then validates it [1].", "citations": [1, 2]},
    ])
    out = await run(retriever_returning([hit(1), hit(2), hit(3)]))
    assert not out["refused"] and out["mode"] == "llm"
    assert out["answer"] == "Annotate it with a type [1]. FastAPI then validates it [2]."  # renumbered by first use
    assert [c["id"] for c in out["citations"]] == ["id2", "id1"]
    assert s.seen == ["grade", "answer"]
    assert out["prompt_tokens"] == 200 and out["completion_tokens"] == 40
    assert [t["step"] for t in out["trace"]] == ["retrieve", "grade", "answer"]


async def test_rewrite_once_then_answer(monkeypatch):
    s = Script(monkeypatch, [
        {"relevant": [], "sufficient": False, "missing": "no typing info"},
        "path parameter type annotation int",
        {"relevant": [1], "sufficient": True, "missing": ""},
        {"sufficient": True, "answer": "Use item_id: int [1].", "citations": [1]},
    ])
    r = retriever_returning([hit(9, 0.2, "CORS")], [hit(1)])
    out = await run(r)
    assert not out["refused"] and out["rewrites"] == 1
    assert r.calls == ["How do I type a path parameter?", "path parameter type annotation int"]
    assert s.seen == ["grade", "rewrite", "grade", "answer"]


async def test_rewrite_then_refuse_when_still_insufficient(monkeypatch):
    Script(monkeypatch, [
        {"relevant": [], "sufficient": False, "missing": "x"},
        "rewritten query",
        {"relevant": [], "sufficient": False, "missing": "x"},
    ])
    out = await run(retriever_returning([hit(1, 0.3, "Other")]), "How do I configure Django migrations?")
    assert out["refused"] and out["answer"] == REFUSAL and out["rewrites"] == 1
    assert out["nearest"][0]["heading_path"] == "Other"
    assert [t["step"] for t in out["trace"]] == ["retrieve", "grade", "rewrite", "retrieve", "grade", "refuse"]


async def test_citation_outside_graded_passages_is_refused(monkeypatch):
    Script(monkeypatch, [
        {"relevant": [1], "sufficient": True, "missing": ""},
        {"sufficient": True, "answer": "Something [3].", "citations": [3]},  # 3 was never graded relevant
    ])
    out = await run(retriever_returning([hit(1), hit(2), hit(3)]))
    assert out["refused"] and out["answer"] == REFUSAL


async def test_answer_without_any_citation_is_refused(monkeypatch):
    Script(monkeypatch, [
        {"relevant": [1], "sufficient": True, "missing": ""},
        {"sufficient": True, "answer": "Trust me.", "citations": []},
    ])
    assert (await run(retriever_returning([hit(1)])))["refused"]


async def test_grader_claiming_sufficient_with_no_relevant_ids_does_not_answer(monkeypatch):
    Script(monkeypatch, [
        {"relevant": [], "sufficient": True, "missing": ""},
        "q2",
        {"relevant": [], "sufficient": True, "missing": ""},
    ])
    assert (await run(retriever_returning([hit(1)])))["refused"]


async def test_llm_error_in_grader_degrades_to_refusal(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)

    async def boom(*a, **k):
        raise TimeoutError("openai slow")

    monkeypatch.setattr(llm, "complete_json", boom)
    monkeypatch.setattr(llm, "complete", boom)
    out = await run(retriever_returning([hit(1)]))
    assert out["refused"]


async def test_offline_mode_uses_score_threshold_and_extractive_answer():
    out = await run(retriever_returning([hit(1, 0.7), hit(2, 0.1)]))
    assert not out["refused"] and out["mode"] == "extractive"
    assert [c["id"] for c in out["citations"]] == ["id1"]
    low = await run(retriever_returning([hit(1, 0.2)]))
    assert low["refused"] and low["rewrites"] == 0  # no rewrite without an LLM


def test_doc_url_builds_public_links():
    assert G.doc_url("fastapi/tutorial/path-params.md", "Path Parameters > Order matters") == \
        "https://fastapi.tiangolo.com/tutorial/path-params/#order-matters"
    assert G.doc_url("pydantic/concepts/models.md", "Models") == "https://docs.pydantic.dev/latest/concepts/models/"
    assert G.doc_url("fastapi/index.md", "FastAPI").startswith("https://fastapi.tiangolo.com/")
