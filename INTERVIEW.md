# DocPilot: defending every design choice

Each section follows the same order: **why it exists → a worked example from this repo → the trade-off**. Every number comes from a script in `eval/` or `bench/`, and the result files are in `bench/results/`.

---

## 1. What problem RAG solves here

**Why.** An LLM answering "how do I forbid extra fields in Pydantic?" from memory may describe the v1 API (`class Config: extra = "forbid"`) or invent an option. RAG fixes this by retrieving the relevant doc sections first and allowing the model to use only those sections. Citations then let a reader check every sentence.

**Example.** The question is embedded, the 5 nearest chunks come back from FAISS, a grader keeps the relevant ones, and the answer cites them, for instance `[1] → pydantic/concepts/models.md › Models › Extra data`.

**Trade-off.** Answers can only be as good as retrieval. If the right section isn't in the top 5, the agent refuses even though the docs contain the answer. That is the false-refusal rate, which `eval/answer_eval.py` reports alongside the refusal rate so neither number looks better than it is.

---

## 2. Chunking: section-aware, 256 tokens with 32 overlap

**Why.** Embeddings work best on text about one thing. A fixed 500-character split cuts mid-sentence and mixes two topics, such as the end of "Path parameters" and the start of "Order matters". So chunks:
1. never cross a heading boundary (each one belongs to exactly one section, which is also what gets cited);
2. pack whole paragraphs and code blocks until the budget is reached;
3. repeat the last 32 tokens of the previous chunk, so a fact split at the boundary can still be found from either side;
4. start with their heading path ("Request Body > Create your data model"), so the vector carries the topic even when the body never names it.

**Example.** The corpus is 179 pages, 1,528 sections and about 400k tokens. That gives **2,632 chunks at 256 tokens** and **1,837 at 512** (`bench/results/chunk_stats.json`).

| chunk size | recall@5 | MRR |
|---|---|---|
| 256 | 0.74 | 0.596 |
| 512 | 0.76 | 0.620 |

**Trade-off.** 512 scored slightly higher, but that is **one question out of 50**, which is noise rather than proof. Smaller chunks give tighter citations and less irrelevant text in the prompt (fewer tokens per answer). Larger chunks keep more context together. I kept 256 as the default for citation precision and prompt cost, and reported both.

**How the eval is fair across chunk sizes.** Gold labels are at **section level** (doc + heading path), not chunk ids, so both chunkings are scored against the same targets. A hit is any top-5 chunk from the gold section.

---

## 3. FAISS vs Chroma behind one interface

**Why.** Both are common choices and the JD names both. A `VectorStore` ABC (`add / search / delete / ids / persist`) means the agent, API and eval never import a backend directly. `VECTOR_BACKEND=chroma` switches it.

**Example (same 2,632 vectors, dim 384; `bench/results/stores.json`).**

| | FAISS `IndexFlatIP` | Chroma (HNSW, cosine) |
|---|---|---|
| build | 0.02 s | 1.54 s |
| query p50 / p95 | 0.064 / 0.069 ms | 0.864 / 0.959 ms |
| disk | 6.8 MB | 31.8 MB |
| recall@5 / MRR | 0.74 / 0.596 | 0.74 / 0.596 (identical) |

**Why identical recall?** FAISS `IndexFlatIP` is *exact* search: it compares the query with every vector. At about 2.6k vectors that takes 64 µs. Chroma uses HNSW, an approximate graph index, but at this size its approximation finds the same top 5. Quality is therefore decided by the embeddings and the chunking, not by the store.

**Trade-off.** FAISS is a library: fast and minimal, but metadata, persistence and deletes are my code (a JSON sidecar plus an `IndexIDMap2`). Chroma is a database: it stores metadata and documents, supports `where` filters and persists on its own, at about 13× the query latency and 4.7× the disk here. At 1M+ vectors I'd switch FAISS to `IndexHNSWFlat` or IVF-PQ, because an exact scan grows linearly.

**Cosine everywhere.** Vectors are L2-normalised once in the embedder, so inner product equals cosine. Chroma's distance is `1 − cos`, so the store returns `1 − d`. Scores mean the same thing on both backends.

---

## 4. Why multiprocessing for ingest, asyncio for I/O

**Why.** These are two kinds of work. Parsing and tokenising Markdown is **CPU-bound pure Python**, and threads would take turns on the GIL, so a `multiprocessing.Pool` uses real cores. Embedding calls to OpenAI, Postgres, Redis and HTTP are **I/O-bound**: the process waits on the network. `asyncio` keeps thousands of waits in flight on one thread with no lock contention.

**Example (`bench/results/pool.json`, 11 workers).**

| corpus | files | serial | pool | speedup |
|---|---|---|---|---|
| 1× (real) | 203 | 0.26 s | 0.42 s | **0.62×** (slower) |
| 4× | 812 | 1.11 s | 0.62 s | 1.8× |
| 16× | 3,248 | 3.89 s | 1.13 s | **3.4×** |

At the real size the pool is *slower*: spawning 11 interpreters costs more than the 0.26 s of work. It starts to pay off at a few hundred files. I kept the pool and documented the crossover rather than hiding it. The CLI accepts `--workers 1` for small corpora.

**Where asyncio shows up.** `Embedder.embed` sends batches concurrently with an `asyncio.Semaphore` (OpenAI provider). The local torch model runs in `asyncio.to_thread` so the event loop keeps serving requests. `/summarize` runs its map steps with `asyncio.gather`. `/ask/stream` pushes graph events through an `asyncio.Queue` to an SSE response.

**Trade-off.** `spawn` workers re-import modules, which is safe but costs startup. Threads plus `to_thread` are used only for blocking calls into C code (torch, FAISS) that release the GIL.

---

## 5. Profiling: one real bottleneck, fixed and measured

**Method.** cProfile over a cold ingest (`bench/profile_ingest.py`), then a median of 3 timed runs.

**Finding.** 98.7% of the time was the embedding forward pass, mostly `torch._C._nn.linear`, which is matrix multiplies. Their cost scales with **batch × padded length**. A transformer pads every text in a batch to the longest one. My embedder sliced batches of 64 in *document order*, so a batch mixed 20-token and 256-token chunks. Measured: batches carried **1.45×** the real token count, which means 45% of the work was multiplying padding.

**Fix.** Sort pending texts by length before batching, then write results back by original index (`docpilot/embed/base.py`). Padding drops to **1.02×**. A test checks that sorted and unsorted runs return identical vectors in the same order.

**Result.** Cold ingest went from **23.43 s to 18.89 s** (median of 3), about 19% faster. Embedding alone went from 23.12 s to 18.57 s.

**Interview follow-up.** "Why not just raise the batch size?" A bigger batch makes padding worse (longer max per batch) and uses more memory. Sorting fixes the waste itself.

---

## 6. Caching: what is keyed, and how

**Why.** Popular questions repeat, and an `/ask` miss costs retrieval plus 2-3 LLM calls (seconds and money). A hit costs one Redis `GET`.

**Answer key** (`docpilot/cache/keys.py`):
```
ask:sha256( endpoint | normalise(question) | backend | chunk_tokens | embed_model | chat_model | prompt_version )
```
- `normalise` lowercases text, collapses whitespace and strips trailing `?!.`, so "What is CORS?" and "what is cors" share one entry.
- Every setting that can change the answer is in the key. Switching FAISS to Chroma, or bumping the prompt version, can never serve an answer built under the old configuration. No manual invalidation is needed.

**Embedding cache.** The key is `emb:sha256(model:text)`, so re-embedding an identical query is free.

**Fallback.** `FallbackCache` tries Redis first. On a connection error it logs once, serves from an in-memory **TTL + LRU** cache (an `OrderedDict`: `move_to_end` on read, `popitem(last=False)` to evict, both O(1)), and **doesn't retry Redis for 15 s**. Without the cooldown, every request would wait out a connect timeout while Redis is down. Tests cover it at two levels: a unit test with a Redis that raises on every call (fallback, cooldown, recovery), and an integration test pointing the cache at a dead port. A manual check stops the Redis container under the running stack and `/ask` keeps answering.

**Trade-off.** Exact-match keys miss paraphrases ("forbid extra fields" vs "reject unknown attributes"). A semantic cache (nearest cached question above a similarity threshold) would catch those, but it risks returning a near-duplicate answer to a different question. For a docs bot that must be right, exact keys are the safer default.

---

## 7. Why LangGraph for /ask

**Why.** The logic isn't one prompt. It's a small state machine with a loop:

```
retrieve → grade ─ sufficient ───────────→ answer ─ citations valid ─→ done
              ├─ insufficient, 0 rewrites → rewrite → retrieve
              └─ insufficient, rewrote once ───────────────→ refuse
```

LangGraph makes the control flow **code, not hope**. "Rewrite at most once, then refuse" is a conditional edge reading `state["rewrites"]`, not an instruction a model may ignore. Each node is a small async function that can be unit-tested on its own. Streaming node updates (`astream(stream_mode="updates")`) is what drives the live agent trace in the UI.

**Refusal is the default.** The answer only goes out if:
1. the grader (strict JSON schema) marks it `sufficient` **and** lists at least one relevant passage;
2. the answer cites at least one passage;
3. every cited number is one the grader marked relevant.

Any failure leads to `refuse`. An LLM error also leads to `refuse`, never to an uncited guess.

**Example.** "How do I set up Django REST framework permissions?" Retrieve returns the closest FastAPI security sections, grade finds none relevant, the query is rewritten, the new retrieval is still irrelevant, and the agent refuses, showing the nearest sections so the user can see what was checked.

**Trade-off.** Grading costs an extra LLM call per question, and the rewrite path adds two more. A single "answer or say you don't know" prompt is cheaper but much less reliable at refusing. The grader is where precision comes from.

---

## 8. PostgreSQL schema (no ORM)

**Why.** The query log answers operational questions directly in SQL: p50/p95 latency split by cache hit or miss, refusal rate, token spend. `percentile_cont(...) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE cache_hit)` computes it in one pass.

**Tables.** `documents` (one row per page) → `chunks` (FK with `ON DELETE CASCADE`, `UNIQUE (content_hash, chunk_tokens)` so re-ingest is an idempotent `ON CONFLICT DO NOTHING`) and `query_log` (latency, cache_hit, refused, rewrites, tokens, `top_k_ids TEXT[]`).

**Indexes, each for a real query.**
- `query_log (created_at DESC)`: `/metrics` windows and "recent queries". An integration test runs `EXPLAIN` to check the planner uses it.
- `query_log (endpoint, cache_hit, created_at DESC)`: latency split by cache outcome.
- Partial index `WHERE refused`: refusals are rare, so the index stays tiny.
- `chunks (document_id, heading_path, chunk_index)`: "all chunks of a doc in order" for `/summarize`.

**Bulk insert.** Chunks go in with one `INSERT … SELECT * FROM unnest($1::text[], …)`, which is one round trip instead of 2,632.

**Trade-off.** Hand-written SQL means hand-maintained migrations (here, idempotent `CREATE … IF NOT EXISTS`). For a three-table service, that's simpler than an ORM and every query stays visible.

---

## 9. Incremental ingest with content hashes

**Why.** Re-embedding unchanged text wastes time (locally) or money (OpenAI).

**How.** A chunk id is `sha256(normalised text)[:16]`, so identical text means an identical id. On re-ingest:
- `new = wanted − existing`: only these are embedded;
- `stale = existing − wanted`: deleted from the store.

**Example.** Re-ingesting the unchanged corpus took **0.4 s vs 27.7 s** cold (2,632 skipped, 0 new). After a parser fix changed some heading paths, the next ingest embedded only the 1,073 changed chunks and removed 1,082 stale ones.

**Trade-off.** Because the heading path is part of the chunk text, renaming a heading re-embeds that section. That's deliberate: the heading is part of what the vector means.

---

## 10. Structured outputs for grading and extraction

**Why.** Free-text model output has to be parsed and can drift. With OpenAI `response_format={"type":"json_schema","strict":true}`, the decoder is constrained to the schema, so `json.loads` can't fail on shape. This is the same pattern I used in Sanjeevani: enums constrain choices, and a fallback runs when validation fails.

**Example.** `/extract` with the `endpoint` schema returns `{method, path, parameters[{name, location ∈ {path, query, header, …}, type, required, default}], …}`. Unknown facts are `null` (nullable types in the schema), so the model has a valid way to say "not stated" instead of inventing a value.

---

## 11. Testing strategy

- **Hermetic by default** (`pytest -q`, about 2 s): the provider key is blanked in `conftest.py`. There is a deterministic hashing `FakeEmbedder`, `fakeredis` and an in-memory repo. The LLM seam (`agent/llm.py: complete / complete_json`) is the single function patched to script each graph branch.
- **Every agent branch is tested:** answer, rewrite then answer, rewrite then refuse, citation outside the graded set leads to refusal, no citation leads to refusal, an LLM exception leads to refusal, and offline threshold mode.
- **Integration** (`pytest -m integration`): the real Postgres (schema, percentiles, `EXPLAIN` uses the index, idempotent chunk upsert) and the real Redis, plus fallback when Redis is unreachable.

---

## 12. Things I'd do next (and why I didn't yet)

- **Hybrid retrieval (BM25 + vectors).** Exact identifiers like `url_too_long` are where dense embeddings miss most. I expect a recall gain; it's next on the list.
- **A stronger embedder** (`bge-small` or OpenAI `text-embedding-3-small`). The OpenAI row in the eval is one flag away: `python -m eval.retrieval_eval --providers local,openai`.
- **A reranker** (a cross-encoder over the top 20). It costs about 50-100 ms per query and usually improves MRR the most.
- **Eval-set caveat.** The 50 labels were written by an AI assistant, and every label is backed by a verbatim evidence quote that a script checks against the gold section (`eval/LABELLING.md`). That's a single annotator with section-level gold. Treat small differences (±0.02) as noise.
