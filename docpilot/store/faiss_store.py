"""FAISS backend: exact inner-product search (IndexFlatIP) wrapped in an IndexIDMap2.

Exact search is right at this scale (thousands of vectors): it's sub-millisecond
and has recall 1.0 by construction, so retrieval quality is measured on the
embeddings and chunking, not on ANN approximation error. Chunk ids (hex content
hashes) map to int64 FAISS ids. Metadata and text sit in a JSON sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

try:  # macOS: faiss and torch each bundle libomp; loading faiss's first segfaults torch (exit 139).
    import torch  # noqa: F401  load torch's OpenMP first, then faiss reuses the loaded runtime
except ImportError:  # API image without torch (OpenAI embeddings only)
    pass
import faiss
import numpy as np

from docpilot.store.base import Hit, VectorStore

# Two OpenMP runtimes (torch's and faiss's) in one process crash when both spin up
# thread pools. Queries here are single-vector exact scans over a few thousand rows
# (well under 1 ms), so FAISS gains nothing from threads: pin it to one.
faiss.omp_set_num_threads(1)


def _int_id(hex_id: str) -> int:
    return int(hex_id[:15], 16)  # 60 bits: fits a signed int64


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.index: faiss.Index | None = None
        self.meta: dict[str, dict] = {}  # hex id -> {meta..., "text": ...}
        self._by_int: dict[int, str] = {}
        if (self.path / "index.faiss").exists():
            self.index = faiss.read_index(str(self.path / "index.faiss"))
            self.meta = json.loads((self.path / "meta.json").read_text())
            self._by_int = {_int_id(h): h for h in self.meta}

    def _ensure(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, chunks, vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._ensure(vectors.shape[1])
        ids = np.array([_int_id(c.id) for c in chunks], dtype=np.int64)
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)
        for c in chunks:
            self.meta[c.id] = {**c.meta(), "text": c.text}
            self._by_int[_int_id(c.id)] = c.id

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(np.asarray(vector, np.float32).reshape(1, -1), k)
        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            h = self._by_int[int(i)]
            m = dict(self.meta[h])
            hits.append(Hit(h, float(s), m, m.pop("text")))
        return hits

    def ids(self) -> set[str]:
        return set(self.meta)

    def delete(self, ids: list[str]) -> None:
        if self.index is None or not ids:
            return
        self.index.remove_ids(np.array([_int_id(h) for h in ids], dtype=np.int64))
        for h in ids:
            self.meta.pop(h, None)
            self._by_int.pop(_int_id(h), None)

    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def persist(self) -> None:
        if self.index is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        (self.path / "meta.json").write_text(json.dumps(self.meta))

    def all_meta(self) -> list[dict]:
        return [{k: v for k, v in m.items() if k != "text"} for m in self.meta.values()]

    def get(self, ids: list[str]) -> list[Hit]:
        out = []
        for h in ids:
            if h in self.meta:
                m = dict(self.meta[h])
                out.append(Hit(h, 1.0, m, m.pop("text")))
        return out

    def by_doc(self, doc_path: str, heading_prefix: str | None = None) -> list[Hit]:
        rows = [(h, m) for h, m in self.meta.items() if m["doc_path"] == doc_path
                and (heading_prefix is None or m["heading_path"].startswith(heading_prefix))]
        rows.sort(key=lambda r: (r[1]["heading_path"], r[1]["chunk_index"]))
        return [Hit(h, 1.0, {k: v for k, v in m.items() if k != "text"}, m["text"]) for h, m in rows]
