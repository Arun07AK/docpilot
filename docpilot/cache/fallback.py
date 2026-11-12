"""Redis cache with an automatic in-memory fallback.

Every call tries Redis first. On a connection error the cache flips to the
in-memory LRU, logs once, and stops trying Redis for `cooldown_s` seconds, so a
dead Redis costs one failed call per cooldown instead of a timeout on every
request. After the cooldown the next call probes Redis again and recovers on its own.

Cache keys (built in docpilot.cache.keys):
  answer:<sha256(normalised question | backend | chunk cfg | embed model | chat model | prompt ver)>
  emb:<sha256(embed model : text)>
"""

from __future__ import annotations

import time

import structlog

from docpilot.cache.memory import MemoryCache

log = structlog.get_logger()


class FallbackCache:
    def __init__(self, redis_client, memory: MemoryCache, default_ttl: int = 3600,
                 cooldown_s: float = 15.0, clock=time.monotonic):
        self.redis = redis_client
        self.memory = memory
        self.default_ttl = default_ttl
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._down_until = 0.0

    @property
    def backend(self) -> str:
        return "memory" if (self.redis is None or self.clock() < self._down_until) else "redis"

    def _fail(self, exc: Exception) -> None:
        if self.clock() >= self._down_until:
            log.warning("cache.redis_down", error=type(exc).__name__, fallback="memory",
                        retry_in_s=self.cooldown_s)
        self._down_until = self.clock() + self.cooldown_s

    async def _try(self, op, *args):
        if self.backend == "memory":
            return False, None
        try:
            return True, await op(*args)
        except Exception as exc:  # ConnectionError, TimeoutError, redis.exceptions.*
            self._fail(exc)
            return False, None

    async def get(self, key: str) -> bytes | None:
        ok, v = await self._try(self.redis.get, key) if self.redis else (False, None)
        return v if ok else await self.memory.get(key)

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl
        ok, _ = await self._try(self.redis.set, key, value, ttl) if self.redis else (False, None)
        if not ok:
            await self.memory.set(key, value, ttl)

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        if not keys:
            return []
        ok, v = await self._try(self.redis.mget, keys) if self.redis else (False, None)
        return list(v) if ok else await self.memory.get_many(keys)

    async def set_many(self, items: dict[str, bytes], ttl: int | None = None) -> None:
        ttl = ttl or self.default_ttl

        async def pipe(items):
            p = self.redis.pipeline(transaction=False)
            for k, v in items.items():
                p.set(k, v, ex=ttl)
            return await p.execute()

        ok, _ = await self._try(pipe, items) if self.redis else (False, None)
        if not ok:
            await self.memory.set_many(items, ttl)

    async def delete_prefix(self, prefix: str) -> int:
        n = await self.memory.delete_prefix(prefix)

        async def scan_del(prefix):
            c = 0
            async for k in self.redis.scan_iter(match=prefix + "*", count=500):
                await self.redis.delete(k)
                c += 1
            return c

        ok, v = await self._try(scan_del, prefix) if self.redis else (False, None)
        return n + (v or 0)

    async def ping(self) -> bool:
        ok, v = await self._try(self.redis.ping) if self.redis else (False, None)
        return bool(ok and v)


def make_cache(url: str, memory_size: int, ttl: int) -> FallbackCache:
    import redis.asyncio as aioredis

    client = aioredis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)
    return FallbackCache(client, MemoryCache(memory_size), ttl)
