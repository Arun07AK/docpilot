"""Both vector backends behind one interface, and the async batched embedder."""

from __future__ import annotations

import numpy as np
import pytest

from docpilot.cache.memory import MemoryCache
from docpilot.embed.base import normalise
from docpilot.ingest.chunk import Chunk
from docpilot.store.chroma_store import ChromaStore
from docpilot.store.faiss_store import FaissStore


def _chunk(i: int, doc="a.md", heading="A > B") -> Chunk:
    return Chunk(f"{i:016x}", doc, "A", heading, i, f"text {i}", 3, f"{i:064x}")


@pytest.fixture(params=["faiss", "chroma"])
def store(request, tmp_path):
    return FaissStore(tmp_path / "f") if request.param == "faiss" else ChromaStore(tmp_path / "c")


def _vecs(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return normalise(rng.normal(size=(n, dim)))


def test_add_search_returns_exact_match_first_with_cosine_score(store):
    v = _vecs(20)
    store.add([_chunk(i) for i in range(20)], v)
    hits = store.search(v[7], 3)
    assert hits[0].id == f"{7:016x}"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].text == "text 7" and hits[0].meta["heading_path"] == "A > B"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_delete_and_ids_and_count(store):
    v = _vecs(5)
    store.add([_chunk(i) for i in range(5)], v)
    store.delete([f"{1:016x}", f"{3:016x}"])
    assert store.count() == 3
    assert store.ids() == {f"{i:016x}" for i in (0, 2, 4)}
    assert all(h.id not in {f"{1:016x}", f"{3:016x}"} for h in store.search(v[1], 5))


def test_by_doc_orders_sections_and_filters_by_heading(store):
    v = _vecs(4)
    cs = [_chunk(0, "x.md", "X > b"), _chunk(1, "x.md", "X > a"), _chunk(2, "y.md", "Y"), _chunk(3, "x.md", "X > a")]
    store.add(cs, v)
    got = store.by_doc("x.md")
    assert [h.meta["heading_path"] for h in got] == ["X > a", "X > a", "X > b"]
    assert len(store.by_doc("x.md", "X > b")) == 1
    assert len(store.all_meta()) == 4


def test_faiss_persists_and_reloads(tmp_path):
    s = FaissStore(tmp_path / "f")
    v = _vecs(6)
    s.add([_chunk(i) for i in range(6)], v)
    s.persist()
    r = FaissStore(tmp_path / "f")
    assert r.count() == 6 and r.search(v[2], 1)[0].id == f"{2:016x}"


def test_empty_store_search_is_empty(store):
    assert store.search(_vecs(1)[0], 5) == []


async def test_embedder_batches_and_normalises(fake_embedder):
    out = await fake_embedder.embed([f"word{i} alpha" for i in range(10)])
    assert out.shape == (10, 64)
    assert fake_embedder.calls == 3  # batch_size=4 -> 4+4+2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


async def test_embedder_cache_skips_already_embedded_texts(fake_embedder):
    fake_embedder.cache = MemoryCache()
    await fake_embedder.embed(["alpha", "beta"])
    before = fake_embedder.texts_embedded
    out = await fake_embedder.embed(["alpha", "beta", "gamma"])
    assert fake_embedder.texts_embedded - before == 1  # only "gamma" hit the model
    assert out.shape == (3, 64)


async def test_embedder_respects_concurrency_limit():
    import asyncio

    from tests.conftest import FakeEmbedder

    class Slow(FakeEmbedder):
        live = peak = 0

        async def _embed_batch(self, texts):
            Slow.live += 1
            Slow.peak = max(Slow.peak, Slow.live)
            await asyncio.sleep(0.01)
            Slow.live -= 1
            return await super()._embed_batch(texts)

    e = Slow(batch_size=1, concurrency=3)
    await e.embed([f"t{i}" for i in range(12)])
    assert Slow.peak == 3


async def test_length_sorted_batching_returns_same_vectors_in_original_order():
    from tests.conftest import FakeEmbedder

    texts = ["a " * n + f"tag{n}" for n in (50, 1, 30, 2, 80, 5, 9)]
    plain = FakeEmbedder(batch_size=2)
    plain.sort_by_length = False
    sorted_ = FakeEmbedder(batch_size=2)
    sorted_.sort_by_length = True
    assert np.allclose(await plain.embed(texts), await sorted_.embed(texts))
