"""Runtime settings. Every knob that changes retrieval or caching lives here.

Values come from the environment (or .env). Keys that change answers (backend,
chunk size, models, prompt version) also feed the cache key, so changing a
setting can never serve a stale cached answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str = ""
    embed_provider: Literal["local", "openai"] = "local"
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    embed_concurrency: int = 4

    vector_backend: Literal["faiss", "chroma"] = "faiss"
    chunk_tokens: int = 256
    chunk_overlap: int = 32
    top_k: int = 5

    chat_model: str = "gpt-4o-mini"
    prompt_version: str = "v1"

    database_url: str = "postgresql://docpilot:docpilot@localhost:5432/docpilot"
    redis_url: str = "redis://localhost:6379/0"
    answer_ttl_s: int = 3600
    embed_ttl_s: int = 7 * 24 * 3600
    memory_cache_size: int = 2048

    data_dir: Path = ROOT / "data"
    ingest_workers: int = 0  # 0 = os.cpu_count()

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    def index_dir(self, backend: str | None = None, chunk_tokens: int | None = None,
                  provider: str | None = None) -> Path:
        b = backend or self.vector_backend
        c = chunk_tokens or self.chunk_tokens
        p = provider or self.embed_provider
        return self.data_dir / "index" / f"{b}-{p}-{c}"


settings = Settings()
