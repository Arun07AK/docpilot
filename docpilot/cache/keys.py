"""Cache key construction, kept in one place so it is easy to reason about.

The answer key folds in every setting that can change the answer. Flip the
backend, the chunk size, a model or the prompt version and the key changes, so
the cache can never return an answer built under a different configuration.
Normalising the question (case, whitespace, trailing ?) lets trivially
different phrasings share an entry without any fuzzy matching.
"""

from __future__ import annotations

import hashlib
import re


def normalise_question(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip().lower())
    return q.rstrip("?!. ")


def answer_key(question: str, *, backend: str, chunk_tokens: int, embed_model: str,
               chat_model: str, prompt_version: str, endpoint: str = "ask") -> str:
    raw = "|".join([endpoint, normalise_question(question), backend, str(chunk_tokens),
                    embed_model, chat_model, prompt_version])
    return f"{endpoint}:" + hashlib.sha256(raw.encode()).hexdigest()
