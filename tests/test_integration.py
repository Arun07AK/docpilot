"""Integration tests against the real Postgres and Redis from docker compose.

Run: docker compose up -d postgres redis && pytest -m integration
"""

from __future__ import annotations

import uuid

import pytest

from docpilot.cache.fallback import make_cache
from docpilot.config import settings
from docpilot.db.repo import Repo
from docpilot.ingest.chunk import Chunk

pytestmark = pytest.mark.integration


async def test_postgres_schema_query_log_and_percentiles():
    repo = Repo(settings.database_url)
    await repo.connect()  # applies schema.sql (idempotent)
    await repo.connect()
    rid = uuid.uuid4().hex[:8]
    for ms, hit in [(100.0, False), (300.0, False), (1.0, True)]:
        await repo.log_query(request_id=rid, endpoint="ask", question="itest", latency_ms=ms, cache_hit=hit,
                             top_k_ids=["a", "b"], backend="faiss")
    m = await repo.metrics(1)
    assert m["queries"] >= 3 and m["p50_hit_ms"] is not None
    recent = await repo.recent(3)
    assert recent[0]["request_id"] == rid
    async with repo.pool.acquire() as con:
        plan = "\n".join(r[0] for r in await con.fetch("EXPLAIN SELECT * FROM query_log ORDER BY created_at DESC LIMIT 5"))
        await con.execute("DELETE FROM query_log WHERE question = 'itest'")
    assert "query_log_created_at_idx" in plan
    await repo.close()


async def test_postgres_chunk_upsert_is_idempotent():
    repo = Repo(settings.database_url)
    await repo.connect()
    tag = uuid.uuid4().hex[:6]
    cs = [Chunk(f"{tag}{i:010x}", f"itest/{tag}.md", "T", "T > S", i, "x", 3, f"{tag}{i:058x}") for i in range(3)]
    n1 = await repo.save_chunks(cs, 999)
    n2 = await repo.save_chunks(cs, 999)
    assert n1 == n2 >= 3
    async with repo.pool.acquire() as con:
        await con.execute("DELETE FROM documents WHERE path LIKE 'itest/%'")  # cascades to chunks
    await repo.close()


async def test_real_redis_roundtrip_and_fallback_to_memory_on_bad_url():
    c = make_cache(settings.redis_url, 16, 60)
    await c.set("itest:k", b"v")
    assert await c.ping() and await c.get("itest:k") == b"v" and c.backend == "redis"
    await c.delete_prefix("itest:")
    dead = make_cache("redis://localhost:1/0", 16, 60)
    await dead.set("k", b"v")
    assert dead.backend == "memory" and await dead.get("k") == b"v"
