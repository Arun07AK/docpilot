# DocPilot: plan

## Context
For the Tredence Summer 2027 AI Engineering Intern application. The JD asks for RAG (chunking, embedding, vector indexing, retrieval), FAISS/Chroma, async FastAPI, OpenAI generation/summarization/extraction, PostgreSQL SQL, Redis/in-memory caching, asyncio/threading/multiprocessing, profiling, tests, logging and small agent workflows. Arun has none of the RAG, vector DB or Redis work yet. DocPilot makes those CV claims true, measured and defensible. It is a FastAPI service that answers questions over the **FastAPI + Pydantic docs** with citations, and says "insufficient evidence" instead of guessing.

**Decisions made:** corpus = FastAPI + Pydantic Markdown docs (MIT). Runtime = Colima via Homebrew. Eval labels = written by Claude, with no review from Arun (see the eval-set section for how they stay verifiable).

**Reused from `~/sbi/sanjeevani`:**
- `api/llm.py`: one `_complete(messages, schema)` seam that tests patch, strict `json_schema` response format, a model constant, and lazy client/dotenv loading. DocPilot gets an async version of the same seam.
- `tests/conftest.py`: an autouse fixture that sets provider keys to `""`, so the suite stays hermetic.
- The module docstring style, the `.gitignore` layout and the slim Dockerfile shape.

## Stack
Python 3.12 (3.13 if all wheels exist) · FastAPI + uvicorn · `openai` async client · `sentence-transformers` (`all-MiniLM-L6-v2`, CPU) as the free default embedder, with OpenAI `text-embedding-3-small` as a config switch · `faiss-cpu` + `chromadb` (persistent client) · `asyncpg` with hand-written SQL files (no ORM) · `redis.asyncio` · `langgraph` · `structlog` (JSON logs, request-id contextvar) · pytest + pytest-asyncio + fakeredis · `pyinstrument`/`cProfile` + `snakeviz` output.

## Layout (`~/docpilot`, git repo, never pushed until Arun says so)
```
docpilot/
  config.py            settings (pydantic-settings): EMBED_PROVIDER, VECTOR_BACKEND, CHUNK_TOKENS...
  ingest/  fetch.py (pinned git checkouts of fastapi + pydantic docs) · parse.py (md -> sections by heading path)
           chunk.py (section-aware, token windows + overlap) · pipeline.py (multiprocessing Pool, content hashes)
  embed/   base.py (async Embedder protocol, batching) · local.py · openai_.py
  store/   base.py (VectorStore interface: add, search, delete_by_ids) · faiss_store.py · chroma_store.py
  db/      schema.sql · indexes.sql · repo.py (asyncpg, raw SQL)
  cache/   base.py · redis_cache.py · memory.py (TTL+LRU) · fallback.py (Redis -> memory on error)
  agent/   graph.py (LangGraph) · prompts.py · llm.py (_complete seam, JSON schema)
  api/     main.py (lifespan, middleware) · routes.py · logging.py
  eval/    questions.jsonl · ooc.jsonl · verify_labels.py · retrieval_eval.py · answer_eval.py · latency_bench.py
  bench/   store_bench.py · profile_ingest.py · results/*.json
  tests/   unit + api (hermetic) · integration (marked, needs compose)
  Dockerfile · docker-compose.yml (api, postgres:16, redis:7) · .env.example
  PLAN.md · STATUS.md (RESUME HERE ledger) · INTERVIEW.md · README.md
```

## Phases (I stop at every boundary for your review)

### Phase 0: setup (short; needs you twice)
- `git init ~/docpilot`, venv, `.gitignore` (with `.env` excluded **before** the first commit), PLAN.md, STATUS.md.
- **You:** run `brew install colima docker docker-compose && colima start`. I can run it if you approve. It installs outside the repo, so I ask first.
- **You:** paste the key into `.env` yourself with one command I give you (`open -e ~/docpilot/.env`). I never read, print or commit it. Only Phases 1b and 3 need it.

### Phase 1: ingest, chunk, embed, FAISS/Chroma, eval set
1. Fetch: pin the fastapi and pydantic repos at specific tags (sparse checkout of `docs/en/docs` and `docs/`), and record the commit SHAs in `data/SOURCES.md`.
2. Parse: strip front-matter, admonitions and code-include directives (`{* ../../docs_src/... *}`, which are inlined from source where feasible). Split by heading path (`H1 > H2 > H3`).
3. Chunk: token windows inside a section (never crossing a section boundary), with overlap. Two configs: **256 tokens/32 overlap** and **512/64**. Each chunk carries `doc_path`, `heading_path`, `chunk_index` and `sha256(normalised text)`.
4. Pipeline: `multiprocessing.Pool` (CPU-bound parse + tokenise) over files. Re-ingest compares content hashes and skips unchanged chunks. Verified by re-running: `0 new, N skipped`.
5. Embed: an async batched embedder (configurable batch size, `asyncio.Semaphore` for concurrency, retry with backoff for OpenAI). The local model runs in a thread pool so it doesn't block the event loop.
6. Vector stores: one `VectorStore` interface. FAISS uses `IndexFlatIP` on normalised vectors plus an id map, persisted to disk. Chroma uses a persistent collection with cosine distance.
7. **Eval set:** ≥45 in-corpus questions, each with `gold_doc`, `gold_heading_path` and a verbatim `evidence` quote, plus ≥15 out-of-corpus questions (Django, Flask, unrelated libraries, near-misses such as features FastAPI doesn't have). `verify_labels.py` asserts that every evidence quote appears in its gold section. This stands in for a human review, and INTERVIEW.md states plainly how the labels were made.
8. `retrieval_eval.py`: recall@5 and MRR at the **section level** (a hit means any top-k chunk is from the gold section), so 256 and 512 are compared fairly. Grid: {FAISS, Chroma} × {256, 512} × {local embedder}, plus one OpenAI-embedder row if the key is present.
- **Verify:** pytest for parse/chunk/hash/stores, label verifier passes, `eval/results/retrieval.json` + a printed table. INTERVIEW.md started (chunk size, FAISS vs Chroma, multiprocessing).

### Phase 2: FastAPI, Postgres, Redis, logging, tests
- `schema.sql`: `documents(id, path, title, sha, ingested_at)`, `chunks(id, document_id FK, heading_path, chunk_index, content_hash UNIQUE, token_count, chunk_size_cfg)`, `query_log(id, request_id, endpoint, question, latency_ms, cache_hit, prompt_tokens, completion_tokens, top_k_ids int[], refused, created_at)`. Indexes on `chunks(document_id)`, `query_log(created_at)` and a partial index on `query_log(cache_hit)`. Applied on startup, idempotently.
- Cache: key = `sha256(normalised question + backend + chunk cfg + model + prompt version)`, so a config change can't serve a stale answer. Embedding cache key = `sha256(provider:model:text)`. TTLs are configurable. `FallbackCache` catches Redis connection errors, logs once, serves from the in-memory TTL+LRU cache, and retries Redis after a cooldown.
- Endpoints (all async): `POST /ingest` → 202 + job id (FastAPI BackgroundTasks; the multiprocessing pool runs off-loop via `run_in_executor`), `GET /ingest/{id}`; `POST /ask` (plain RAG here, replaced by the agent in Phase 3); `GET /metrics` (SQL aggregates: count, p50/p95 via `percentile_cont`, cache-hit rate, tokens); `GET /healthz`.
- Logging: middleware sets an `X-Request-ID` contextvar, and structlog JSON lines carry it through retrieval, cache and LLM calls.
- Tests: hermetic (fake embedder, fakeredis, mocked `_complete`, repo layer faked). Integration tests (`-m integration`) run against compose Postgres + Redis, including a test that kills Redis and confirms the fallback.
- **Verify:** `pytest -q` green offline; `docker compose up` → curl `/ask` twice and see `cache_hit` flip in `query_log`; stop the redis container and `/ask` still works.

### Phase 3: LangGraph agent, /summarize, /extract
- Graph: `retrieve → grade` (an LLM grader with a strict JSON schema: `{relevant_ids[], sufficient: bool}`) → `answer` (cites `[doc#heading]` and only uses graded chunks) | `rewrite` (once, guarded by a state counter) → `retrieve` | `refuse` ("insufficient evidence", plus the closest sections found). The answer step's citations are validated against the retrieved ids, and any made-up citation leads to a refusal.
- `/summarize` (a doc path or heading path; map-reduce for long docs) and `/extract` (a caller picks a named schema, e.g. `endpoint_params`: method, path, parameters[], response_model, via strict JSON schema).
- Model: a `CHAT_MODEL` constant (gpt-4o-mini or the cheapest current equivalent, checked against your key's model list at the start of this phase).
- **Verify:** graph unit tests for every branch with a mocked `_complete` (answer, rewrite then answer, rewrite then refuse, bad citation then refuse); manual `/ask` on 5 real questions.

### Phase 4: benchmarks, profiling, README, CV
- `answer_eval.py`: % of out-of-corpus questions correctly refused, and % of in-corpus questions answered (false-refusal rate reported too, so the number is honest).
- `latency_bench.py`: async httpx, 100 cold (unique, cache flushed) vs 100 warm `/ask` calls → p50/p95 for each.
- `store_bench.py`: FAISS vs Chroma, index build time, query p50/p95 and disk size.
- `profile_ingest.py`: cProfile the full ingest, name the top bottleneck **from the profile** (no guessing in advance), fix it, re-run 3× and report the median before/after.
- README: architecture diagram (Mermaid), setup, numbers table taken from `bench/results/*.json`. INTERVIEW.md finished (cache keying, why LangGraph, multiprocessing vs asyncio, each written as why → worked example → trade-off).
- CV: fill the RAG slot as "DocPilot" with 2 to 3 bullets that only use measured numbers, add FAISS, Chroma, Redis and RAG to Skills, remove ParkSense, run `tectonic -X compile` and confirm 1 page (`pdfinfo` page count). Text rules: no em dashes, none of the banned words, no "not X, it's Y". **Note:** the GitHub link stays dead until you say push.

## Rough API cost (OpenAI)
- Embeddings (only if you use the OpenAI embedder row): the corpus is roughly 1 to 2M tokens × 2 chunk configs at $0.02/1M ≈ **$0.05**.
- Agent (`/ask` ≈ 2 to 3 calls, ~3k input tokens): ≈ $0.001 per question. Eval (60 questions × ~3 runs) + latency bench (100 cold) + manual testing ≈ **$0.50 to $1.50**.
- **Total expected under $2.** I'll stop and ask if spend looks set to pass $5.

## Resume / ledger
STATUS.md always holds: current phase, last verified command and its result, the next step, and `cd ~/docpilot && claude` + "read STATUS.md and continue". Long runs (embeddings, benchmarks) write progress to `bench/results/` so they can resume.

## Verification overall
Each phase ends with a pytest run (output pasted), the phase's command outputs, and a STATUS.md update, and then I stop for your review.
