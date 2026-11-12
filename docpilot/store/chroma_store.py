"""Chroma backend: a persistent collection with an HNSW index and cosine space.

We pass our own embeddings (Chroma never calls a model). Chroma returns a cosine
*distance* d = 1 - cos, so score = 1 - d keeps both stores on the same scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docpilot.store.base import Hit, VectorStore

_ADD_BATCH = 4000  # below Chroma's max batch size


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, path: Path, collection: str = "docpilot"):
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path),
                                                settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks, vectors: np.ndarray) -> None:
        for i in range(0, len(chunks), _ADD_BATCH):
            part = chunks[i:i + _ADD_BATCH]
            self.col.upsert(
                ids=[c.id for c in part],
                embeddings=vectors[i:i + _ADD_BATCH].tolist(),
                documents=[c.text for c in part],
                metadatas=[c.meta() for c in part],
            )

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self.col.count() == 0:
            return []
        r = self.col.query(query_embeddings=[np.asarray(vector, np.float32).tolist()], n_results=k,
                           include=["metadatas", "documents", "distances"])
        return [Hit(i, 1.0 - float(d), m, doc) for i, d, m, doc in
                zip(r["ids"][0], r["distances"][0], r["metadatas"][0], r["documents"][0])]

    def ids(self) -> set[str]:
        return set(self.col.get(include=[])["ids"])

    def delete(self, ids: list[str]) -> None:
        for i in range(0, len(ids), _ADD_BATCH):
            self.col.delete(ids=ids[i:i + _ADD_BATCH])

    def count(self) -> int:
        return self.col.count()

    def persist(self) -> None:
        pass  # PersistentClient writes through

    def all_meta(self) -> list[dict]:
        return self.col.get(include=["metadatas"])["metadatas"]

    def get(self, ids: list[str]) -> list[Hit]:
        if not ids:
            return []
        r = self.col.get(ids=ids, include=["metadatas", "documents"])
        return [Hit(i, 1.0, m, d) for i, m, d in zip(r["ids"], r["metadatas"], r["documents"])]

    def by_doc(self, doc_path: str, heading_prefix: str | None = None) -> list[Hit]:
        r = self.col.get(where={"doc_path": doc_path}, include=["metadatas", "documents"])
        rows = [Hit(i, 1.0, m, d) for i, m, d in zip(r["ids"], r["metadatas"], r["documents"])
                if heading_prefix is None or m["heading_path"].startswith(heading_prefix)]
        rows.sort(key=lambda h: (h.meta["heading_path"], h.meta["chunk_index"]))
        return rows
