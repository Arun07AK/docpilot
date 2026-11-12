"""PostgreSQL access through asyncpg with plain SQL. No ORM; every query is visible.

`Repo` is the single place that talks SQL. The API depends on the `BaseRepo`
protocol, so tests swap in `MemoryRepo` and stay hermetic. If Postgres is down,
the API logs it and serves answers anyway; only the query log is lost.
"""

from __future__ import annotations

import time
from pathlib import Path

SCHEMA = (Path(__file__).parent / "schema.sql").read_text()

METRICS_SQL = """
SELECT
    count(*)                                                        AS queries,
    coalesce(avg(cache_hit::int), 0)                                AS cache_hit_rate,
    coalesce(avg(refused::int), 0)                                  AS refusal_rate,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms)        AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)        AS p95_ms,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE cache_hit)     AS p50_hit_ms,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit) AS p50_miss_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit) AS p95_miss_ms,
    coalesce(sum(prompt_tokens), 0)                                 AS prompt_tokens,
    coalesce(sum(completion_tokens), 0)                             AS completion_tokens
FROM query_log
WHERE created_at > now() - make_interval(hours => $1)
"""

RECENT_SQL = """
SELECT request_id, endpoint, question, latency_ms, cache_hit, refused, rewrites, created_at
FROM query_log ORDER BY created_at DESC LIMIT $1
"""

UPSERT_DOC_SQL = """
INSERT INTO documents (path, title, source) VALUES ($1, $2, $3)
ON CONFLICT (path) DO UPDATE SET title = EXCLUDED.title, ingested_at = now()
RETURNING id
"""

INSERT_CHUNKS_SQL = """
INSERT INTO chunks (id, document_id, heading_path, chunk_index, chunk_tokens, token_count, content_hash)
SELECT * FROM unnest($1::text[], $2::bigint[], $3::text[], $4::int[], $5::int[], $6::int[], $7::text[])
ON CONFLICT (content_hash, chunk_tokens) DO NOTHING
"""

LOG_SQL = """
INSERT INTO query_log (request_id, endpoint, question, latency_ms, cache_hit, refused, rewrites,
                       prompt_tokens, completion_tokens, top_k_ids, backend)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""


class BaseRepo:
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def log_query(self, **row) -> None: ...
    async def metrics(self, hours: int = 24) -> dict: ...
    async def recent(self, limit: int = 20) -> list[dict]: ...
    async def save_chunks(self, chunks, chunk_tokens: int) -> int: ...
    async def healthy(self) -> bool: ...


class Repo(BaseRepo):
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None

    async def connect(self) -> None:
        import asyncpg

        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10, command_timeout=5)
        async with self.pool.acquire() as con:
            await con.execute(SCHEMA)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def healthy(self) -> bool:
        try:
            async with self.pool.acquire() as con:
                return await con.fetchval("SELECT 1") == 1
        except Exception:
            return False

    async def log_query(self, **r) -> None:
        async with self.pool.acquire() as con:
            await con.execute(LOG_SQL, r["request_id"], r["endpoint"], r["question"], r["latency_ms"],
                              r["cache_hit"], r.get("refused", False), r.get("rewrites", 0),
                              r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
                              r.get("top_k_ids", []), r["backend"])

    async def metrics(self, hours: int = 24) -> dict:
        async with self.pool.acquire() as con:
            row = await con.fetchrow(METRICS_SQL, hours)
        return {k: (float(v) if v is not None and k not in ("queries",) else v) for k, v in dict(row).items()}

    async def recent(self, limit: int = 20) -> list[dict]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(RECENT_SQL, limit)
        return [{**dict(r), "created_at": r["created_at"].isoformat()} for r in rows]

    async def save_chunks(self, chunks, chunk_tokens: int) -> int:
        """Batch upsert: one round trip per document, then one unnest() insert for all chunks."""
        doc_ids: dict[str, int] = {}
        async with self.pool.acquire() as con:
            async with con.transaction():
                for path, title in {c.doc_path: c.title for c in chunks}.items():
                    doc_ids[path] = await con.fetchval(UPSERT_DOC_SQL, path, title, path.split("/")[0])
                cols = list(zip(*[(c.id, doc_ids[c.doc_path], c.heading_path, c.chunk_index,
                                   chunk_tokens, c.token_count, c.content_hash) for c in chunks]))
                if cols:
                    await con.execute(INSERT_CHUNKS_SQL, *[list(col) for col in cols])
                return await con.fetchval("SELECT count(*) FROM chunks WHERE chunk_tokens = $1", chunk_tokens)


class MemoryRepo(BaseRepo):
    """In-process stand-in used by tests and when Postgres is unreachable."""

    def __init__(self):
        self.rows: list[dict] = []
        self.chunks: dict[str, dict] = {}

    async def healthy(self) -> bool:
        return True

    async def log_query(self, **r) -> None:
        self.rows.append({**r, "created_at": time.time()})

    async def metrics(self, hours: int = 24) -> dict:
        import numpy as np

        lat = [r["latency_ms"] for r in self.rows]
        hit = [r["latency_ms"] for r in self.rows if r["cache_hit"]]
        miss = [r["latency_ms"] for r in self.rows if not r["cache_hit"]]
        pct = lambda xs, q: float(np.percentile(xs, q)) if xs else None
        n = len(self.rows)
        return {"queries": n,
                "cache_hit_rate": sum(r["cache_hit"] for r in self.rows) / n if n else 0.0,
                "refusal_rate": sum(r.get("refused", False) for r in self.rows) / n if n else 0.0,
                "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "p50_hit_ms": pct(hit, 50),
                "p50_miss_ms": pct(miss, 50), "p95_miss_ms": pct(miss, 95),
                "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in self.rows),
                "completion_tokens": sum(r.get("completion_tokens", 0) for r in self.rows)}

    async def recent(self, limit: int = 20) -> list[dict]:
        return [{k: r.get(k) for k in ("request_id", "endpoint", "question", "latency_ms",
                                         "cache_hit", "refused", "rewrites")} for r in self.rows[::-1][:limit]]

    async def save_chunks(self, chunks, chunk_tokens: int) -> int:
        for c in chunks:
            self.chunks[c.id] = {"doc": c.doc_path, "tokens": chunk_tokens}
        return len(self.chunks)
