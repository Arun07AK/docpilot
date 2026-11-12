"""Shared fixtures. The suite is hermetic: no OpenAI key, no network, no model downloads.

Same idea as Sanjeevani's conftest: blank the provider key so every LLM path
takes its offline branch unless a test scripts the LLM seam explicitly.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from docpilot.config import settings
from docpilot.embed.base import Embedder


@pytest.fixture(autouse=True)
def _no_real_providers(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""

    name = "fake"
    dim = 64

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self.texts_embedded = 0

    async def _embed_batch(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z_]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(batch_size=4, concurrency=2)


MINI_DOCS = {
    "fastapi/docs/en/docs/tutorial/path-params.md": """# Path Parameters

You can declare path parameters with the same syntax used by Python format strings.

## Path parameters with types

You can declare the type of a path parameter in the function, using standard Python type annotations.
In this case, item_id is declared to be an int.

```python
# this comment is not a heading
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

## Order matters { #order-matters }

Because path operations are evaluated in order, you need to declare /users/me before /users/{user_id}.

/// tip

Otherwise the path for /users/{user_id} would match also for /users/me.

///
""",
    "fastapi/docs/en/docs/tutorial/cors.md": """# CORS

## Use CORSMiddleware

Import CORSMiddleware and add it with allow_origins to permit cross origin requests from browsers.
""",
    "pydantic/docs/concepts/models.md": """## Basic model usage

Models are classes that inherit from BaseModel and define fields as annotated attributes.

## Extra data

By default extra attributes are ignored. Set extra='forbid' in model_config to raise a validation error.

!!! note "Strictness"
    Validation in lax mode coerces strings like '123' to int.
""",
}


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for rel, text in MINI_DOCS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "fastapi" / "docs_src").mkdir(parents=True, exist_ok=True)
    return root
