"""FastAPI app: async endpoints, SSE agent trace, background ingest, SQL-backed metrics.

`create_app(svc)` takes prebuilt services so tests inject fakes. In production
the lifespan builds them: embedder, vector store, Redis-with-fallback cache and
the Postgres repo (falling back to an in-memory repo if Postgres is down, so
answers keep flowing while only the query log degrades).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docpilot.agent import llm
from docpilot.agent.prompts import EXTRACT_SCHEMAS
from docpilot.api.logging import RequestIdMiddleware, configure_logging
from docpilot.config import settings
from docpilot.service import DocPilot, Result

log = structlog.get_logger()
WEB = Path(__file__).resolve().parents[2] / "web"


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    use_cache: bool = True


class SummarizeIn(BaseModel):
    doc_path: str
    heading: str | None = None
    use_cache: bool = True


class ExtractIn(BaseModel):
    schema_name: str = Field(alias="schema")
    doc_path: str | None = None
    heading: str | None = None
    query: str | None = None
    use_cache: bool = True

    model_config = {"populate_by_name": True}


class IngestIn(BaseModel):
    workers: int | None = None


async def build_services() -> DocPilot:
    from docpilot.cache.fallback import make_cache
    from docpilot.db.repo import MemoryRepo, Repo
    from docpilot.embed.base import get_embedder
    from docpilot.store.base import get_store

    cache = make_cache(settings.redis_url, settings.memory_cache_size, settings.answer_ttl_s)
    embedder = get_embedder(cache=cache)
    if hasattr(embedder, "warm"):
        await asyncio.to_thread(embedder.warm)
    store = await asyncio.to_thread(get_store)
    repo = Repo(settings.database_url)
    try:
        await repo.connect()
    except Exception as exc:
        log.warning("db.unavailable", error=type(exc).__name__, fallback="memory")
        repo = MemoryRepo()
    log.info("startup", backend=store.name, chunks=store.count(), embedder=embedder.name,
             cache=cache.backend, llm=llm.available())
    return DocPilot(store, embedder, cache, repo)


def create_app(svc: DocPilot | None = None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = svc or await build_services()
        app.state.ingest_lock = asyncio.Lock()
        yield
        await app.state.svc.repo.close()

    app = FastAPI(title="DocPilot", version="1.0", lifespan=lifespan,
                  description="Cited answers over the FastAPI and Pydantic docs.")
    app.add_middleware(RequestIdMiddleware)
    if (WEB / "assets").exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    def S(request: Request) -> DocPilot:
        return request.app.state.svc

    async def log_query(request: Request, endpoint: str, text: str, r: Result) -> None:
        try:
            await S(request).repo.log_query(
                request_id=request.state.request_id, endpoint=endpoint, question=text[:1000],
                latency_ms=r.latency_ms, cache_hit=r.cache_hit, refused=r.refused, rewrites=r.rewrites,
                prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                top_k_ids=r.top_k_ids, backend=S(request).store.name)
        except Exception as exc:  # the log must never break an answer
            log.warning("db.log_failed", error=type(exc).__name__)

    def respond(request: Request, r: Result) -> JSONResponse:
        body = {**r.body, "latency_ms": round(r.latency_ms, 1), "request_id": request.state.request_id}
        return JSONResponse(body, headers={"x-cache": "HIT" if r.cache_hit else "MISS"})

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/healthz")
    async def healthz(request: Request):
        s = S(request)
        return {"ok": True, "backend": s.store.name, "chunks": s.store.count(), "embedder": s.embedder.name,
                "chunk_tokens": settings.chunk_tokens, "cache": s.cache.backend,
                "redis": await s.cache.ping(), "postgres": await s.repo.healthy(),
                "llm": llm.available(), "chat_model": settings.chat_model if llm.available() else None}

    # -------------------------------------------------------------------- ask
    @app.post("/ask")
    async def ask(body: AskIn, request: Request):
        r = await S(request).ask(body.question, body.use_cache)
        await log_query(request, "ask", body.question, r)
        return respond(request, r)

    @app.post("/ask/stream")
    async def ask_stream(body: AskIn, request: Request):
        """Server-Sent Events: one `step` event per graph node, then a `result` event."""
        q: asyncio.Queue = asyncio.Queue()

        async def on_step(step: dict):
            await q.put(("step", step))

        async def run():
            try:
                r = await S(request).ask(body.question, body.use_cache, on_step=on_step)
                await log_query(request, "ask", body.question, r)
                await q.put(("result", {**r.body, "latency_ms": round(r.latency_ms, 1),
                                        "request_id": request.state.request_id}))
            except Exception as exc:
                log.exception("ask.failed")
                await q.put(("error", {"detail": type(exc).__name__}))
            await q.put(None)

        task = asyncio.create_task(run())

        async def events():
            try:
                while (item := await q.get()) is not None:
                    yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    # ------------------------------------------------------ summarize/extract
    @app.post("/summarize")
    async def summarize(body: SummarizeIn, request: Request):
        try:
            r = await S(request).summarize(body.doc_path, body.heading, body.use_cache)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'"))
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        await log_query(request, "summarize", f"{body.doc_path} > {body.heading or ''}", r)
        return respond(request, r)

    @app.get("/extract/schemas")
    async def extract_schemas():
        return {k: {"description": v["description"], "schema": v["schema"]} for k, v in EXTRACT_SCHEMAS.items()}

    @app.post("/extract")
    async def extract(body: ExtractIn, request: Request):
        try:
            r = await S(request).extract(body.schema_name, body.doc_path, body.heading, body.query, body.use_cache)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        except ValueError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        await log_query(request, "extract", body.query or f"{body.doc_path} > {body.heading or ''}", r)
        return respond(request, r)

    @app.get("/documents")
    async def documents(request: Request):
        return await asyncio.to_thread(S(request).list_docs)

    # ----------------------------------------------------------------- ingest
    @app.post("/ingest", status_code=202)
    async def ingest(body: IngestIn, request: Request, bg: BackgroundTasks):
        s = S(request)
        if request.app.state.ingest_lock.locked():
            raise HTTPException(409, "an ingest job is already running")
        job_id = uuid.uuid4().hex[:10]
        s.jobs[job_id] = {"id": job_id, "status": "queued", "stage": None, "progress": {}}
        bg.add_task(_run_ingest, request.app, job_id, body.workers)
        return {"job_id": job_id, "status_url": f"/ingest/{job_id}"}

    @app.get("/ingest/{job_id}")
    async def ingest_status(job_id: str, request: Request):
        job = S(request).jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return job

    # ---------------------------------------------------------------- metrics
    @app.get("/metrics")
    async def metrics(request: Request, hours: int = 24):
        s = S(request)
        try:
            m = await s.repo.metrics(hours)
            recent = await s.repo.recent(12)
        except Exception as exc:
            raise HTTPException(503, f"metrics unavailable: {type(exc).__name__}")
        return {"window_hours": hours, **m, "recent": recent, "cache_backend": s.cache.backend,
                "backend": s.store.name, "chunks": s.store.count()}

    @app.post("/cache/flush")
    async def flush(request: Request):
        s = S(request)
        n = 0
        for prefix in ("ask:", "summarize:", "extract:"):
            n += await s.cache.delete_prefix(prefix)
        return {"flushed": n}

    return app


async def _run_ingest(app: FastAPI, job_id: str, workers: int | None) -> None:
    from docpilot.ingest.pipeline import ingest

    s: DocPilot = app.state.svc
    job = s.jobs[job_id]

    def progress(stage: str, info: dict):
        job.update(status="running", stage=stage)
        job["progress"].update(info)

    async with app.state.ingest_lock:
        try:
            stats = await ingest(s.store, s.embedder, workers=workers, on_progress=progress,
                                 on_chunks=lambda chunks: s.repo.save_chunks(chunks, settings.chunk_tokens))
            job.update(status="done", stage="done", result={
                "files": stats.files, "sections": stats.sections, "chunks": stats.chunks, "new": stats.new,
                "skipped": stats.skipped, "removed": stats.removed,
                "timings": {k: round(v, 2) for k, v in stats.timings.items()}})
        except Exception as exc:
            log.exception("ingest.failed")
            job.update(status="failed", error=type(exc).__name__)


app = create_app()
