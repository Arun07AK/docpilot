# STATUS: RESUME HERE

**Resume:** `cd ~/docpilot && claude`, then say "read STATUS.md and continue".

## Current
- Phase: scaffold pushed; building Phase 1 (ingest, chunk, embed, stores, eval set).
- Mode: Arun asked for an end-to-end build without stopping at phase boundaries (13 Nov 2026).
- Corpus: FastAPI + Pydantic Markdown docs. Runtime: Colima. Eval labels: written by Claude and checked by `eval/verify_labels.py`.
- The CV (`~/AEKAY/05-CAREER/CV/Arun_AK_CV_Tredence_AIEng.tex`) is owned by the "Resume tailoring for job application" session. Send it measured numbers; don't edit the CV from here.

## Phases
| # | Phase | Status |
|---|---|---|
| 0 | Repo, venv, Colima, scaffold, public push | in progress |
| 1 | Ingest, chunk, embed, FAISS/Chroma, eval set | pending |
| 2 | FastAPI, Postgres, Redis, logging, tests | pending |
| 3 | LangGraph agent, /summarize, /extract | pending |
| 4 | Web UI, benchmarks, profiling, README, INTERVIEW | pending |

## Needs Arun
- `OPENAI_API_KEY` in `~/docpilot/.env` (run `open -e ~/docpilot/.env`). Needed for the agent and for answer and latency numbers.

## Measured numbers
None yet.
