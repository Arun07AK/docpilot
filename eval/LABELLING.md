# How the eval set was labelled

**Files:** `questions.jsonl` (50 in-corpus), `ooc.jsonl` (20 out-of-corpus). Every label points at `sections.jsonl` (1528 parsed sections, FastAPI 0.162.5 + Pydantic v2.14.3).

## Method (in-corpus)
1. Browsed sections doc by doc, with code blocks hidden, and kept those where a short prose passage answers a concrete developer question.
2. For each candidate, grepped every section for the answer's key terms (parameter names, numbers, error codes). If another section also answered it, the candidate was dropped. Examples dropped this way: `convert_underscores` (also in header-param-models), `Depends(scope="function")` (also in advanced-dependencies), `FailFast` (also in types), `check_fields` (three sections), `__modify_schema__` (also in migration).
3. `evidence` was cut programmatically from the section text using start/end anchors, so it is an exact substring (40-250 chars) that contains the answer.
4. Questions were written in a developer's own words. A script flagged any 3-word overlap between a question and its evidence or heading, and the flagged questions were reworded.
5. Each row also has a `difficulty` field (easy/medium/hard). This field is not in the original schema.

## Spread
- Source: 30 FastAPI, 20 Pydantic, across 42 docs (no doc used more than 3 times).
- FastAPI: tutorial 14, advanced 9, how-to 4, deployment 1, CLI/async 2 (`fastapi-cli.md`, `async.md`).
- Pydantic: concepts 14, errors 3 (1 usage error, 2 validation-error codes), migration 1, version policy 2.
- Difficulty: 15 easy, 25 medium, 10 hard. Hard questions need a specific detail, such as a default value (GZip `minimum_size`=500, `RedirectResponse` 307), a parameter name (`root_path_in_servers`, `alias_priority=2`), a version plus status code (401 since 0.122.0), an error code (`date_from_datetime_inexact`, URL limit 2083) or a benchmark figure (~2.5x).

## Out-of-corpus
- 10 `unrelated`: Django migrations, Flask blueprints, React, Kubernetes HPA, PostgreSQL VACUUM, GitHub Actions, Rust, Terraform, pandas, Spring Boot.
- 10 `near_miss`: built-in features that don't exist (rate limiting, cron scheduler, BrotliMiddleware, permessage-deflate, gRPC router, circuit breaker, msgspec backend), numeric benchmarks against named tools (vs Litestar, vs attrs), and version trivia (the date Python 3.8 support was dropped).
- Check: each near-miss's key terms were grepped across all sections (`rate.?limit`, `cron`, `brotli`, `deflate`, `gRPC`, `msgspec`, `circuit breaker`, `Litestar`, `Python 3\.8`, `attrs`, and others). Each search returned 0 hits, or only unrelated hits. Examples of unrelated hits: TechEmpower is mentioned, but with no numbers and no Litestar. `attrs` appears only inside a config-rename list. "autoscaling" appears only in the FastAPI Cloud blurb.

## Known limitations
- Labels come from a single annotator, an AI assistant (Claude). No human reviewed them. `verify_labels.py`-style checks only prove that each evidence string exists in its gold section, not that the question is natural or the answer complete.
- Gold labels are section-level, with exactly one gold section per question. A retriever that returns another section which also partly answers a question is scored as a miss. Uniqueness was checked with keyword grep, which can miss paraphrases.
- Heading paths are copied exactly as the parser produced them, including its quirks: dropped leading backticks, and every Pydantic error heading nested under its doc's first heading ("Class not fully defined > ..." for usage errors, "arguments_type > ..." for validation errors).
- Near-miss "unanswerable" is judged from keyword absence. A model may still answer partially from neighbouring sections (for example GZip for the Brotli question), which is the behaviour these questions exist to test.
