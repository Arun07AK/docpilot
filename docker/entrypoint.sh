#!/bin/sh
# First boot: fetch the pinned corpus and build the configured index. Later boots: incremental (no-op).
set -e
python -m docpilot.cli fetch
python -m docpilot.cli ingest --backend "${VECTOR_BACKEND:-faiss}" --chunk "${CHUNK_TOKENS:-256}" --provider "${EMBED_PROVIDER:-local}"
exec "$@"
