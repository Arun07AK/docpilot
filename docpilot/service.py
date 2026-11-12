"""Application services: retrieval, ask, summarize, extract. One object built at startup.

The API layer is thin: it parses the request, calls a method here, and logs the
query. Everything here is async; CPU-bound or blocking work (torch, FAISS
search, the ingest process pool) runs off the event loop via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import structlog

from docpilot.agent import llm
from docpilot.agent.graph import _cites, build_graph, doc_url
from docpilot.agent.prompts import EXTRACT_SCHEMAS, EXTRACT_SYSTEM, SUMMARY_SCHEMA, SUMMARY_SYSTEM
from docpilot.cache.keys import answer_key
from docpilot.config import settings

log = structlog.get_logger()

MAP_TOKENS = 3000  # per map step in map-reduce summarization


@dataclass
class Result:
    body: dict
    cache_hit: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    refused: bool = False
    rewrites: int = 0
    top_k_ids: list[str] = field(default_factory=list)


class DocPilot:
    def __init__(self, store, embedder, cache, repo):
        self.store, self.embedder, self.cache, self.repo = store, embedder, cache, repo
        self.graph = build_graph(self.retrieve)
        self.jobs: dict[str, dict] = {}
        # what POST /ingest builds: the configured chunking over the configured corpus
        self.ingest_cfg = {"chunk_tokens": settings.chunk_tokens, "overlap": settings.chunk_overlap,
                           "corpus_dir": None}

    # ------------------------------------------------------------- retrieval
    async def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        vec = await self.embedder.embed_one(query)
        hits = await asyncio.to_thread(self.store.search, vec, k or settings.top_k)
        return [{"id": h.id, "score": h.score, "text": h.text, **h.meta} for h in hits]

    def _key(self, endpoint: str, text: str) -> str:
        return answer_key(text, endpoint=endpoint, backend=self.store.name, chunk_tokens=settings.chunk_tokens,
                          embed_model=self.embedder.name, chat_model=settings.chat_model if llm.available() else "none",
                          prompt_version=settings.prompt_version)

    async def _cached(self, endpoint: str, text: str):
        raw = await self.cache.get(self._key(endpoint, text))
        return json.loads(raw) if raw else None

    async def _store(self, endpoint: str, text: str, body: dict) -> None:
        await self.cache.set(self._key(endpoint, text), json.dumps(body).encode(), settings.answer_ttl_s)

    # ------------------------------------------------------------------- ask
    async def ask(self, question: str, use_cache: bool = True, on_step=None) -> Result:
        t0 = time.perf_counter()
        if use_cache and (body := await self._cached("ask", question)):
            body["cached"] = True
            return Result(body, True, (time.perf_counter() - t0) * 1000, refused=body["refused"],
                          top_k_ids=[c["id"] for c in body["citations"]])
        init = {"question": question, "rewrites": 0, "trace": [], "prompt_tokens": 0, "completion_tokens": 0}
        state: dict = dict(init)
        async for update in self.graph.astream(init, stream_mode="updates"):
            for node, delta in update.items():
                state.update(delta or {})
                if on_step and delta and delta.get("trace"):
                    await on_step(delta["trace"][-1])
        body = {"question": question, "answer": state.get("answer", ""), "refused": bool(state.get("refused")),
                "citations": state.get("citations", []), "nearest": state.get("nearest", []),
                "mode": state.get("mode", "llm" if llm.available() else "extractive"),
                "rewrites": state.get("rewrites", 0), "trace": state.get("trace", []), "cached": False,
                "usage": {"prompt_tokens": state.get("prompt_tokens", 0),
                          "completion_tokens": state.get("completion_tokens", 0)}}
        if use_cache:
            await self._store("ask", question, body)
        return Result(body, False, (time.perf_counter() - t0) * 1000, body["usage"]["prompt_tokens"],
                      body["usage"]["completion_tokens"], body["refused"], body["rewrites"],
                      [h["id"] for h in state.get("hits", [])])

    # ------------------------------------------------------------- documents
    def list_docs(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for m in self.store.all_meta():
            d = docs.setdefault(m["doc_path"], {"doc_path": m["doc_path"], "title": m["title"], "sections": set()})
            d["sections"].add(m["heading_path"])
        return sorted(({**d, "sections": sorted(d["sections"])} for d in docs.values()),
                      key=lambda d: d["doc_path"])

    async def _section_text(self, doc_path: str, heading: str | None) -> tuple[str, list]:
        hits = await asyncio.to_thread(self.store.by_doc, doc_path, heading)
        if not hits:
            raise KeyError(f"no indexed text for {doc_path}" + (f" > {heading}" if heading else ""))
        return "\n\n".join(h.text for h in hits), hits

    # ------------------------------------------------------------- summarize
    async def summarize(self, doc_path: str, heading: str | None = None, use_cache: bool = True) -> Result:
        t0 = time.perf_counter()
        ck = f"{doc_path}#{heading or ''}"
        if use_cache and (body := await self._cached("summarize", ck)):
            return Result({**body, "cached": True}, True, (time.perf_counter() - t0) * 1000)
        text, hits = await self._section_text(doc_path, heading)
        if not llm.available():
            raise RuntimeError("summarize needs OPENAI_API_KEY")
        usage = llm.Usage()
        groups, cur, size = [], [], 0
        for h in hits:  # map-reduce when the section is long
            tok = int(h.meta.get("token_count", len(h.text) // 4))
            if cur and size + tok > MAP_TOKENS:
                groups.append(cur)
                cur, size = [], 0
            cur.append(h.text)
            size += tok
        groups.append(cur)
        sem = asyncio.Semaphore(4)

        async def summarize_part(part: str) -> dict:
            async with sem:
                out, u = await llm.complete_json([{"role": "system", "content": SUMMARY_SYSTEM},
                                                  {"role": "user", "content": part}], SUMMARY_SCHEMA, "summary")
                usage.add(u)
                return out

        parts = await asyncio.gather(*(summarize_part("\n\n".join(g)) for g in groups))
        if len(parts) > 1:
            merged = "\n\n".join(f"Summary: {p['summary']}\nPoints: " + "; ".join(p["key_points"]) for p in parts)
            final = await summarize_part("Combine these partial summaries of one document:\n\n" + merged)
        else:
            final = parts[0]
        body = {"doc_path": doc_path, "heading": heading, "url": doc_url(doc_path, heading or ""),
                **final, "chunks": len(hits), "map_steps": len(groups), "cached": False,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}}
        if use_cache:
            await self._store("summarize", ck, body)
        return Result(body, False, (time.perf_counter() - t0) * 1000, usage.prompt_tokens, usage.completion_tokens)

    # --------------------------------------------------------------- extract
    async def extract(self, schema_name: str, doc_path: str | None = None, heading: str | None = None,
                      query: str | None = None, use_cache: bool = True) -> Result:
        t0 = time.perf_counter()
        if schema_name not in EXTRACT_SCHEMAS:
            raise KeyError(f"unknown schema {schema_name!r}; choose from {sorted(EXTRACT_SCHEMAS)}")
        ck = f"{schema_name}|{doc_path}|{heading}|{query}"
        if use_cache and (body := await self._cached("extract", ck)):
            return Result({**body, "cached": True}, True, (time.perf_counter() - t0) * 1000)
        if doc_path:
            text, hits = await self._section_text(doc_path, heading)
            sources = _cites([{"id": h.id, "score": 1.0, "text": h.text, **h.meta} for h in hits[:1]])
        elif query:
            found = await self.retrieve(query, k=3)
            text, sources = "\n\n".join(h["text"] for h in found), _cites(found)
        else:
            raise ValueError("give doc_path (+ optional heading) or query")
        if not llm.available():
            raise RuntimeError("extract needs OPENAI_API_KEY")
        spec = EXTRACT_SCHEMAS[schema_name]
        out, u = await llm.complete_json(
            [{"role": "system", "content": f"{EXTRACT_SYSTEM}\nTarget: {spec['description']}"},
             {"role": "user", "content": text[:12000]}], spec["schema"], schema_name, max_tokens=900)
        body = {"schema": schema_name, "data": out, "sources": sources, "cached": False,
                "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}}
        if use_cache:
            await self._store("extract", ck, body)
        return Result(body, False, (time.perf_counter() - t0) * 1000, u.prompt_tokens, u.completion_tokens)
