"""In-memory TTL + LRU cache: the fallback when Redis is down, and the test double.

An OrderedDict gives O(1) get/set/evict. Reads move a key to the end (most
recent), and inserts past `maxsize` pop from the front (least recent). Expired
entries are dropped lazily on read, so there's no background sweeper thread.
"""

from __future__ import annotations

import time
from collections import OrderedDict


class MemoryCache:
    name = "memory"

    def __init__(self, maxsize: int = 2048, clock=time.monotonic):
        self.maxsize = maxsize
        self.clock = clock
        self._d: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    async def get(self, key: str) -> bytes | None:
        item = self._d.get(key)
        if item is None:
            return None
        expires, value = item
        if expires and expires < self.clock():
            del self._d[key]
            return None
        self._d.move_to_end(key)
        return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        self._d[key] = ((self.clock() + ttl) if ttl else 0.0, value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    async def get_many(self, keys: list[str]) -> list[bytes | None]:
        return [await self.get(k) for k in keys]

    async def set_many(self, items: dict[str, bytes], ttl: int | None = None) -> None:
        for k, v in items.items():
            await self.set(k, v, ttl)

    async def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._d if k.startswith(prefix)]
        for k in keys:
            del self._d[k]
        return len(keys)

    async def ping(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._d)
