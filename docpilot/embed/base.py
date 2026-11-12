"""Async, batched embeddings behind one interface.

`Embedder.embed(texts)` splits the input into batches, runs up to `concurrency`
batches at once (an asyncio.Semaphore), and returns L2-normalised float32
vectors, so inner product equals cosine similarity in every store.

An optional cache (get_many/set_many) short-circuits texts embedded before.
That matters at query time: a repeated question costs zero embedding calls.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable

import numpy as np


def normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class Embedder:
    name: str = "base"
    dim: int = 0

    def __init__(self, batch_size: int = 64, concurrency: int = 4, cache=None):
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.cache = cache  # async get_many(keys)->list|None, set_many(dict)

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def cache_key(self, text: str) -> str:
        return "emb:" + hashlib.sha256(f"{self.name}:{text}".encode()).hexdigest()

    async def embed(self, texts: list[str],
                    on_batch: Callable[[int, int], None] | None = None) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = list(range(len(texts)))
        if self.cache is not None and texts:
            keys = [self.cache_key(t) for t in texts]
            cached = await self.cache.get_many(keys)
            todo = []
            for i, hit in enumerate(cached):
                if hit is not None:
                    out[i] = np.frombuffer(hit, dtype=np.float32) if isinstance(hit, bytes) else np.asarray(hit, np.float32)
                else:
                    todo.append(i)

        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        sem = asyncio.Semaphore(self.concurrency)
        done = 0

        async def run(idx: list[int]):
            nonlocal done
            async with sem:
                vecs = normalise(await self._embed_batch([texts[i] for i in idx]))
            for i, v in zip(idx, vecs):
                out[i] = v
            done += len(idx)
            if on_batch:
                on_batch(done, len(todo))
            if self.cache is not None:
                await self.cache.set_many({self.cache_key(texts[i]): v.tobytes() for i, v in zip(idx, vecs)})

        await asyncio.gather(*(run(b) for b in batches))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]


def get_embedder(provider: str | None = None, cache=None) -> Embedder:
    from docpilot.config import settings

    provider = provider or settings.embed_provider
    if provider == "openai":
        from docpilot.embed.openai_ import OpenAIEmbedder

        return OpenAIEmbedder(settings.openai_embed_model, settings.embed_batch_size,
                              settings.embed_concurrency, cache)
    from docpilot.embed.local import LocalEmbedder

    return LocalEmbedder(settings.local_embed_model, settings.embed_batch_size, cache=cache)
