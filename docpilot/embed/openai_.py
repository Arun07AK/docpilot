"""OpenAI embeddings (text-embedding-3-small) with async batching and retry.

Batches of up to 64 texts go out concurrently, at most `concurrency` at a time.
Rate limits and transient errors are retried with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np

from docpilot.embed.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, batch_size: int = 64, concurrency: int = 4, cache=None):
        super().__init__(batch_size=batch_size, concurrency=concurrency, cache=cache)
        self.model = model
        self.name = f"openai:{model}"
        self.dim = 1536
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            from docpilot.config import settings

            self._client = AsyncOpenAI(api_key=settings.openai_api_key or None, timeout=30)
        return self._client

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        delay = 1.0
        for attempt in range(5):
            try:
                resp = await self._get_client().embeddings.create(model=self.model, input=texts)
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(delay + random.random() * 0.5)
                delay *= 2
        raise RuntimeError("unreachable")
