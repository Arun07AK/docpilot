"""Structured JSON logging with a request id on every line.

The middleware reads X-Request-ID (or creates one), binds it into structlog's
contextvars, and echoes it on the response. Contextvars follow the asyncio
task, so log lines from retrieval, the cache and LLM calls deep inside the
request carry the same id without passing it around by hand.
"""

from __future__ import annotations

import logging
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        request.state.request_id = rid
        t = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path.startswith(("/ask", "/summarize", "/extract", "/ingest")):
            structlog.get_logger().info("http.request", method=request.method, path=request.url.path,
                                        status=response.status_code,
                                        ms=round((time.perf_counter() - t) * 1000, 1))
        return response
