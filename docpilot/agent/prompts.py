"""Prompts and JSON schemas. Bump PROMPT_VERSION in config when these change (it's in the cache key)."""

GRADER_SYSTEM = (
    "You grade retrieved documentation passages for a question about FastAPI or Pydantic. "
    "For each numbered passage decide if it contains information that directly helps answer the "
    "question. Then decide if the relevant passages TOGETHER are sufficient to answer the question "
    "without outside knowledge. Be strict: a passage on the same topic that does not contain the "
    "answer is NOT relevant. If the question is about another framework or product, nothing is relevant."
)

GRADER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevant": {"type": "array", "items": {"type": "integer"},
                     "description": "Numbers of passages that directly help answer the question."},
        "sufficient": {"type": "boolean"},
        "missing": {"type": "string", "description": "What is missing, if insufficient. Empty otherwise."},
    },
    "required": ["relevant", "sufficient", "missing"],
}

REWRITE_SYSTEM = (
    "Rewrite the user's question into a short search query for the FastAPI and Pydantic "
    "documentation. Use the vocabulary the docs would use (class names, parameter names, "
    "concept names). Keep the intent. Output only the query."
)

ANSWER_SYSTEM = (
    "You answer questions about FastAPI and Pydantic using ONLY the numbered passages provided. "
    "Rules:\n"
    "1. Every sentence that states a fact ends with a citation like [1] or [2][3], using passage numbers.\n"
    "2. Never use knowledge that is not in the passages. If the passages do not answer the question, "
    "set sufficient=false and leave answer empty.\n"
    "3. Be concise: 2-6 sentences. Include a short code snippet only if a passage contains one that answers the question.\n"
    "4. citations lists every passage number you cited."
)

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sufficient": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["sufficient", "answer", "citations"],
}

SUMMARY_SYSTEM = (
    "Summarize the documentation text for a developer. Use only the text given. "
    "Return a 2-4 sentence summary and 3-6 key points (each one line, concrete: names, defaults, rules)."
)

SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_points"],
}

EXTRACT_SYSTEM = (
    "Extract structured fields from the documentation text. Use only facts stated in the text "
    "or shown in its code. Use null for anything the text does not state. Do not invent values."
)


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


EXTRACT_SCHEMAS: dict[str, dict] = {
    "endpoint": {
        "description": "An HTTP endpoint shown in the docs: method, path, parameters, models.",
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "method": _nullable("string"),
                "path": _nullable("string"),
                "parameters": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"name": {"type": "string"},
                                   "location": {"type": "string", "enum": ["path", "query", "header", "cookie", "body", "form", "file", "unknown"]},
                                   "type": _nullable("string"), "required": _nullable("boolean"),
                                   "default": _nullable("string")},
                    "required": ["name", "location", "type", "required", "default"]}},
                "request_model": _nullable("string"),
                "response_model": _nullable("string"),
                "status_code": _nullable("integer"),
            },
            "required": ["method", "path", "parameters", "request_model", "response_model", "status_code"],
        },
    },
    "config_options": {
        "description": "Configuration options or settings and their defaults.",
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {"options": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "type": _nullable("string"),
                               "default": _nullable("string"), "description": {"type": "string"}},
                "required": ["name", "type", "default", "description"]}}},
            "required": ["options"],
        },
    },
    "error": {
        "description": "An error or exception: what raises it, why, and how to fix it.",
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {"error_type": {"type": "string"}, "raised_when": {"type": "string"},
                           "fix": _nullable("string"), "example_message": _nullable("string")},
            "required": ["error_type", "raised_when", "fix", "example_message"],
        },
    },
}

REFUSAL = "Insufficient evidence: the FastAPI and Pydantic docs I index don't answer this."
