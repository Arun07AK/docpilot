"""In-memory TTL/LRU, Redis with fallback, and cache keys."""

from __future__ import annotations

import fakeredis
import pytest

from docpilot.cache.fallback import FallbackCache
from docpilot.cache.keys import answer_key, normalise_question
from docpilot.cache.memory import MemoryCache


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


async def test_memory_ttl_expires():
    clock = Clock()
    c = MemoryCache(clock=clock)
    await c.set("k", b"v", ttl=10)
    assert await c.get("k") == b"v"
    clock.t += 11
    assert await c.get("k") is None and len(c) == 0


async def test_memory_lru_evicts_least_recently_used():
    c = MemoryCache(maxsize=2)
    await c.set("a", b"1")
    await c.set("b", b"2")
    await c.get("a")  # a is now most recent
    await c.set("c", b"3")
    assert await c.get("b") is None
    assert await c.get("a") == b"1" and await c.get("c") == b"3"


async def test_fallback_uses_redis_when_up():
    r = fakeredis.aioredis.FakeRedis()
    c = FallbackCache(r, MemoryCache())
    await c.set("k", b"v", 60)
    assert await r.get("k") == b"v"
    assert await c.get("k") == b"v" and c.backend == "redis"
    await c.set_many({"x": b"1", "y": b"2"})
    assert await c.get_many(["x", "y", "z"]) == [b"1", b"2", None]


class DeadRedis:
    calls = 0

    async def _boom(self, *a, **k):
        DeadRedis.calls += 1
        raise ConnectionError("redis down")

    get = set = mget = ping = _boom


async def test_fallback_switches_to_memory_and_backs_off_then_recovers():
    clock = Clock()
    DeadRedis.calls = 0
    c = FallbackCache(DeadRedis(), MemoryCache(clock=clock), cooldown_s=15, clock=clock)
    await c.set("k", b"v", 60)  # redis fails -> stored in memory
    assert c.backend == "memory"
    assert await c.get("k") == b"v"
    assert DeadRedis.calls == 1  # cooldown: second call didn't touch redis
    clock.t += 16
    good = fakeredis.aioredis.FakeRedis()
    c.redis = good  # redis came back
    await c.set("k2", b"v2", 60)
    assert c.backend == "redis" and await good.get("k2") == b"v2"


async def test_delete_prefix_clears_both_layers():
    r = fakeredis.aioredis.FakeRedis()
    c = FallbackCache(r, MemoryCache())
    await c.set("ask:1", b"a")
    await c.set("emb:1", b"e")
    await c.memory.set("ask:2", b"m")
    assert await c.delete_prefix("ask:") == 2
    assert await c.get("emb:1") == b"e" and await c.get("ask:1") is None


def test_question_normalisation():
    assert normalise_question("  How do   I use CORS?? ") == "how do i use cors"


def test_answer_key_changes_with_every_answer_affecting_setting():
    base = dict(backend="faiss", chunk_tokens=256, embed_model="m", chat_model="c", prompt_version="v1")
    k = answer_key("What is CORS?", **base)
    assert k == answer_key("what is cors", **base)
    for field, val in [("backend", "chroma"), ("chunk_tokens", 512), ("embed_model", "m2"),
                       ("chat_model", "c2"), ("prompt_version", "v2")]:
        assert answer_key("What is CORS?", **{**base, field: val}) != k
    assert answer_key("What is CORS?", endpoint="summarize", **base).startswith("summarize:")
