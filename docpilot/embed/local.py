"""Local sentence-transformers embedder (free, offline after the first download).

`model.encode` is a blocking torch call. It runs in a worker thread via
asyncio.to_thread so the event loop keeps serving requests. Concurrency is 1
because torch already uses every core inside a single call.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from docpilot.embed.base import Embedder

_models: dict[str, object] = {}
_lock = threading.Lock()


def _load(name: str):
    with _lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer

            _models[name] = SentenceTransformer(name, device="cpu")
        return _models[name]


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64, cache=None):
        super().__init__(batch_size=batch_size, concurrency=1, cache=cache)
        self.model_name = model_name
        self.name = f"local:{model_name}"
        self.dim = 384

    def warm(self) -> None:
        m = _load(self.model_name)
        self.dim = m.get_sentence_embedding_dimension()

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = await asyncio.to_thread(_load, self.model_name)
        return await asyncio.to_thread(
            model.encode, texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
