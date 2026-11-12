-- DocPilot schema. Hand-written, idempotent, applied on API startup.
-- Three tables: what was ingested (documents, chunks) and what was asked (query_log).

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    path         TEXT        NOT NULL UNIQUE,          -- "fastapi/tutorial/body.md"
    title        TEXT        NOT NULL,
    source       TEXT        NOT NULL,                 -- fastapi | pydantic
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT        PRIMARY KEY,             -- first 16 hex of content_hash; also the vector id
    document_id   BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading_path  TEXT        NOT NULL,
    chunk_index   INT         NOT NULL,
    chunk_tokens  INT         NOT NULL,                -- chunking config: 256 | 512
    token_count   INT         NOT NULL,
    content_hash  CHAR(64)    NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (content_hash, chunk_tokens)
);

CREATE TABLE IF NOT EXISTS query_log (
    id                 BIGSERIAL   PRIMARY KEY,
    request_id         TEXT        NOT NULL,
    endpoint           TEXT        NOT NULL,           -- ask | summarize | extract
    question           TEXT        NOT NULL,
    latency_ms         REAL        NOT NULL,
    cache_hit          BOOLEAN     NOT NULL,
    refused            BOOLEAN     NOT NULL DEFAULT false,
    rewrites           SMALLINT    NOT NULL DEFAULT 0,
    prompt_tokens      INT         NOT NULL DEFAULT 0,
    completion_tokens  INT         NOT NULL DEFAULT 0,
    top_k_ids          TEXT[]      NOT NULL DEFAULT '{}',
    backend            TEXT        NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Join path chunk -> document, and "all chunks of a doc" for /summarize.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id, heading_path, chunk_index);
-- /metrics windows by time; newest-first listing in the UI.
CREATE INDEX IF NOT EXISTS query_log_created_at_idx ON query_log (created_at DESC);
-- Latency split by cache outcome: the planner can use (endpoint, cache_hit) directly.
CREATE INDEX IF NOT EXISTS query_log_endpoint_hit_idx ON query_log (endpoint, cache_hit, created_at DESC);
-- Refusals are rare; a partial index keeps it tiny.
CREATE INDEX IF NOT EXISTS query_log_refused_idx ON query_log (created_at DESC) WHERE refused;
