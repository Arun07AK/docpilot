"""HTTP layer: every endpoint, the SSE stream, cache headers, request ids and the query log."""

from __future__ import annotations

import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

from docpilot.agent import llm
from docpilot.api.main import create_app
from docpilot.cache.fallback import FallbackCache
from docpilot.cache.memory import MemoryCache
from docpilot.db.repo import MemoryRepo
from docpilot.ingest.pipeline import ingest
from docpilot.service import DocPilot
from docpilot.store.faiss_store import FaissStore


@pytest.fixture
def client(mini_corpus, tmp_path, fake_embedder, monkeypatch):
    import asyncio

    store = FaissStore(tmp_path / "idx")
    asyncio.run(ingest(store, fake_embedder, 64, 8, workers=1, corpus_dir=mini_corpus))
    cache = FallbackCache(fakeredis.aioredis.FakeRedis(), MemoryCache())
    svc = DocPilot(store, fake_embedder, cache, MemoryRepo())
    svc.ingest_cfg = {"chunk_tokens": 64, "overlap": 8, "corpus_dir": mini_corpus}
    with TestClient(create_app(svc)) as c:
        c.svc = svc
        yield c


def test_healthz_reports_components(client):
    h = client.get("/healthz").json()
    assert h["ok"] and h["backend"] == "faiss" and h["chunks"] > 0
    assert h["redis"] is True and h["postgres"] is True and h["llm"] is False


def test_ask_miss_then_hit_and_query_log(client):
    q = {"question": "how do I declare the type of a path parameter"}
    r1 = client.post("/ask", json=q, headers={"x-request-id": "abc123"})
    assert r1.status_code == 200 and r1.headers["x-cache"] == "MISS"
    assert r1.headers["x-request-id"] == "abc123" and r1.json()["request_id"] == "abc123"
    b1 = r1.json()
    assert b1["mode"] == "extractive" and b1["citations"]
    assert b1["citations"][0]["url"].startswith("https://")
    r2 = client.post("/ask", json={"question": "How do I declare the TYPE of a path parameter?"})
    assert r2.headers["x-cache"] == "HIT" and r2.json()["cached"] is True
    assert r2.json()["answer"] == b1["answer"]
    rows = client.svc.repo.rows
    assert [r["cache_hit"] for r in rows] == [False, True]
    assert rows[0]["top_k_ids"] and rows[0]["backend"] == "faiss"


def test_ask_validation_error(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422


def test_ask_stream_emits_steps_then_result(client):
    with client.stream("POST", "/ask/stream", json={"question": "cors middleware allow origins", "use_cache": False}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [blk.split("\n", 1) for blk in body.strip().split("\n\n")]
    names = [e[0].removeprefix("event: ") for e in events]
    assert names[0] == "step" and names[-1] == "result"
    steps = [json.loads(e[1].removeprefix("data: "))["step"] for e in events if e[0] == "event: step"]
    assert steps[0] == "retrieve" and steps[1] == "grade"
    result = json.loads(events[-1][1].removeprefix("data: "))
    assert "answer" in result and "latency_ms" in result


def test_unrelated_question_is_refused_offline(client):
    b = client.post("/ask", json={"question": "kubernetes horizontal pod autoscaler yaml"}).json()
    assert b["refused"] is True and b["citations"] == []


def test_summarize_and_extract_need_llm_and_validate_input(client):
    assert client.post("/summarize", json={"doc_path": "nope.md"}).status_code == 404
    assert client.post("/summarize", json={"doc_path": "fastapi/tutorial/cors.md"}).status_code == 503
    assert client.post("/extract", json={"schema": "bogus", "query": "x"}).status_code == 404
    assert client.post("/extract", json={"schema": "endpoint"}).status_code == 422
    assert set(client.get("/extract/schemas").json()) == {"endpoint", "config_options", "error"}


def test_summarize_and_extract_with_scripted_llm(client, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    outs = [{"summary": "CORS lets browsers call your API.", "key_points": ["Add CORSMiddleware"]},
            {"method": "GET", "path": "/items/{item_id}", "parameters": [
                {"name": "item_id", "location": "path", "type": "int", "required": True, "default": None}],
             "request_model": None, "response_model": None, "status_code": None}]

    async def complete_json(messages, schema, name="output", max_tokens=700):
        return outs.pop(0), llm.Usage(10, 5)

    monkeypatch.setattr(llm, "complete_json", complete_json)
    s = client.post("/summarize", json={"doc_path": "fastapi/tutorial/cors.md"}).json()
    assert s["summary"].startswith("CORS") and s["map_steps"] == 1 and s["url"].startswith("https://")
    e = client.post("/extract", json={"schema": "endpoint", "doc_path": "fastapi/tutorial/path-params.md",
                                      "heading": "Path Parameters > Path parameters with types"}).json()
    assert e["data"]["parameters"][0]["name"] == "item_id" and e["sources"]
    again = client.post("/summarize", json={"doc_path": "fastapi/tutorial/cors.md"})
    assert again.headers["x-cache"] == "HIT"


def test_documents_lists_docs_with_sections(client):
    docs = client.get("/documents").json()
    paths = {d["doc_path"] for d in docs}
    assert "fastapi/tutorial/path-params.md" in paths
    pp = next(d for d in docs if d["doc_path"] == "fastapi/tutorial/path-params.md")
    assert "Path Parameters > Order matters" in pp["sections"]


def test_ingest_runs_in_background_and_reports_incremental_result(client):
    r = client.post("/ingest", json={"workers": 1})
    assert r.status_code == 202
    job = client.get(r.json()["status_url"]).json()
    assert job["status"] == "done", job  # TestClient runs background tasks before returning
    assert job["result"]["new"] == 0 and job["result"]["skipped"] > 0
    assert client.get("/ingest/missing").status_code == 404


def test_metrics_and_cache_flush(client):
    client.post("/ask", json={"question": "cors middleware"})
    client.post("/ask", json={"question": "cors middleware"})
    m = client.get("/metrics").json()
    assert m["queries"] == 2 and m["cache_hit_rate"] == 0.5
    assert m["p50_hit_ms"] is not None and m["p50_miss_ms"] >= m["p50_hit_ms"]
    assert len(m["recent"]) == 2
    assert client.post("/cache/flush").json()["flushed"] >= 1
    assert client.post("/ask", json={"question": "cors middleware"}).headers["x-cache"] == "MISS"


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "DocPilot" in r.text
