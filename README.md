# DocPilot

A production-style RAG and agent backend that answers questions over the **FastAPI and Pydantic documentation**, with a citation for every claim. When the retrieved evidence doesn't support an answer, it replies "insufficient evidence" instead of guessing.

> **Status:** in active development. Endpoint and benchmark sections are filled in as each part lands and gets measured. Numbers in this README come only from scripts in `eval/` and `bench/`.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Ingestion (multiprocessing pool)"]
        A[FastAPI + Pydantic docs<br/>pinned git tags] --> B[Parse Markdown<br/>by heading path]
        B --> C[Section-aware chunking<br/>token windows + overlap]
        C --> D{content hash<br/>seen before?}
        D -- no --> E[Batched async embeddings<br/>local or OpenAI]
        D -- yes --> X[skip]
    end
    E --> F[(VectorStore interface<br/>FAISS | Chroma)]
    C --> P[(PostgreSQL<br/>documents, chunks, query_log)]

    U[Client / Web UI] --> API[Async FastAPI]
    API --> R{Redis cache<br/>in-memory fallback}
    R -- hit --> U
    R -- miss --> G
    subgraph G["LangGraph agent"]
        G1[retrieve] --> G2[grade relevance]
        G2 -- sufficient --> G3[answer with citations]
        G2 -- weak, first try --> G4[rewrite query] --> G1
        G2 -- still weak --> G5[refuse: insufficient evidence]
    end
    G1 --> F
    API --> P
```

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI (async), uvicorn |
| LLM | OpenAI chat with strict JSON-schema structured outputs |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) or OpenAI `text-embedding-3-small`, chosen by config |
| Vector store | FAISS (`IndexFlatIP`) and Chroma behind one `VectorStore` interface, chosen by config |
| Agent | LangGraph: retrieve, grade, then answer, rewrite once or refuse |
| Database | PostgreSQL 16 via `asyncpg`, hand-written SQL schema and indexes (no ORM) |
| Cache | Redis 7 for answers and embeddings, with an in-memory TTL/LRU fallback |
| Concurrency | `multiprocessing` for CPU-bound parse and chunk, `asyncio` for I/O |
| Observability | structlog JSON logs carrying a per-request id; `/metrics` from SQL aggregates |
| Tests | pytest, hermetic (LLM and embeddings mocked, fakeredis) plus marked integration tests |
| Packaging | Dockerfile + Docker Compose (api, postgres, redis) |

## Endpoints (planned)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Start ingestion as a background job; returns a job id |
| `GET` | `/ingest/{id}` | Job progress: files parsed, chunks new and skipped |
| `POST` | `/ask` | Cited answer from the LangGraph agent, or a refusal |
| `POST` | `/summarize` | Summary of a document or one section |
| `POST` | `/extract` | Structured fields from a section via OpenAI JSON-schema output |
| `GET` | `/metrics` | Query count, p50/p95 latency, cache-hit rate, token usage |
| `GET` | `/healthz` | Liveness, plus Postgres and Redis status |

## Evaluation (planned, measured by scripts in `eval/` and `bench/`)

- Retrieval: recall@5 and MRR on 45+ labelled questions, FAISS vs Chroma, 256 vs 512-token chunks.
- Answer behaviour: share of out-of-corpus questions correctly refused, plus the false-refusal rate.
- Latency: `/ask` p50/p95 on a cache miss vs a cache hit.
- Profiling: one cProfile pass over ingestion, with the bottleneck fixed and before/after times reported.

## Run (once the build lands)

```bash
cp .env.example .env      # add OPENAI_API_KEY
docker compose up --build
```

## License

MIT for the code. The corpus is the FastAPI and Pydantic documentation (both MIT); it is fetched at build time and not committed.
