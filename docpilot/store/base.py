"""One interface for vector stores, so FAISS and Chroma swap by config.

Both stores receive pre-normalised vectors and return a similarity score in
[-1, 1] (cosine), so retrieval code and thresholds don't care which one runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    meta: dict
    text: str


class VectorStore(ABC):
    name: str

    @abstractmethod
    def add(self, chunks: list, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[Hit]: ...

    @abstractmethod
    def ids(self) -> set[str]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def all_meta(self) -> list[dict]: ...

    def get(self, ids: list[str]) -> list[Hit]:
        raise NotImplementedError

    def by_doc(self, doc_path: str, heading_prefix: str | None = None) -> list[Hit]:
        raise NotImplementedError


def get_store(backend: str | None = None, path: Path | None = None) -> VectorStore:
    from docpilot.config import settings

    backend = backend or settings.vector_backend
    path = path or settings.index_dir(backend)
    if backend == "chroma":
        from docpilot.store.chroma_store import ChromaStore

        return ChromaStore(path)
    from docpilot.store.faiss_store import FaissStore

    return FaissStore(path)
