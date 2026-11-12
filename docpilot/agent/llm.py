"""The one OpenAI boundary. Every chat call in DocPilot goes through `complete`.

Same pattern as Sanjeevani's llm.py, made async: one seam that tests patch,
strict JSON-schema structured output when a schema is given, and token usage
returned with the content so /metrics and the query log can count spend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from docpilot.config import settings

_client = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


def available() -> bool:
    return bool(settings.openai_api_key)


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=30, max_retries=2)
    return _client


async def complete(messages: list[dict], schema: dict | None = None, name: str = "output",
                   temperature: float = 0.0, max_tokens: int = 700) -> tuple[str, Usage]:
    """Returns (content, usage). With `schema`, content is a JSON string that matches it."""
    kwargs = {"model": settings.chat_model, "messages": messages, "temperature": temperature,
              "max_tokens": max_tokens}
    if schema is not None:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": name, "strict": True, "schema": schema}}
    resp = await _get_client().chat.completions.create(**kwargs)
    u = resp.usage
    return resp.choices[0].message.content or "", Usage(getattr(u, "prompt_tokens", 0) or 0,
                                                      getattr(u, "completion_tokens", 0) or 0)


async def complete_json(messages: list[dict], schema: dict, name: str = "output",
                        max_tokens: int = 700) -> tuple[dict, Usage]:
    content, usage = await complete(messages, schema=schema, name=name, max_tokens=max_tokens)
    return json.loads(content), usage
