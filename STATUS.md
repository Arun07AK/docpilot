# STATUS: RESUME HERE

**Resume:** `cd ~/docpilot && claude`, then say "read STATUS.md and continue".
Before resuming: `colima start` (if the Mac rebooted), then `docker compose up -d postgres redis`.

## Where it stands (13 Nov 2026, ~01:40 IST)
Everything in the plan is built, tested and pushed except the steps that need the OpenAI key.

| # | Phase | Status |
|---|---|---|
| 0 | Repo, venv, Colima, scaffold, public push | done (github.com/Arun07AK/docpilot) |
| 1 | Ingest, chunk, embed, FAISS/Chroma, eval set, retrieval eval | done |
| 2 | FastAPI, Postgres, Redis + fallback, logging, tests | done (50 hermetic + 3 integration pass) |
| 3 | LangGraph agent, /summarize, /extract | built and unit-tested with a scripted LLM; **live run needs the key** |
| 4 | UI, benchmarks, profiling, README, INTERVIEW | done, except the LLM-dependent numbers |

## Blocked on Arun
`OPENAI_API_KEY` in `~/docpilot/.env` (run `open -e ~/docpilot/.env`, paste after `OPENAI_API_KEY=`). Then:
1. Restart the server: `.venv/bin/uvicorn docpilot.api.main:app --port 8000`
2. `python -m eval.answer_eval` → refusal rate and citation accuracy → README table + INTERVIEW
3. `python -m eval.latency_bench` → /ask p50/p95 miss vs hit → README table
4. Optional: `python -m docpilot.cli ingest-all --provider openai && python -m eval.retrieval_eval --providers local,openai`
5. Try /summarize and /extract in the UI; take `docs/screenshot-ask.png` (the README references it)
6. Send the numbers to the "Resume tailoring for job application" session (it owns the CV .tex; don't edit the CV from here)

## Measured numbers (bench/results/*.json)
- Retrieval (50 q, k=5): 256 tok → recall 0.74 / MRR 0.596; 512 tok → 0.76 / 0.620; FAISS = Chroma.
- Stores: FAISS p50 0.064 ms vs Chroma 0.864 ms; build 0.02 s vs 1.54 s; disk 6.8 vs 31.8 MB.
- Profiling: cold ingest 23.43 s → 18.89 s (length-sorted batches; padding 1.45x → 1.02x).
- Pool: 0.62x at 1x corpus, 1.8x at 4x, 3.4x at 16x (11 workers).
- Incremental re-ingest: 0.4 s vs 27.7 s cold.
- Redis-down live check: /ask 200, hits served from memory, auto-recovery after the 15 s cooldown.

## Gotchas found
- macOS: faiss and torch each bundle libomp, which segfaults. Fixed by `scripts/macos_dev_setup.sh` (symlinks faiss to torch's libomp) plus `faiss.omp_set_num_threads(1)`. Linux/Docker is unaffected.
- `/usr/local` Homebrew is Intel (Tier 3, broken Go). Use `/opt/homebrew/bin/brew`.
- `gh` is an x86 build; push with `git -c credential.helper='!gh auth git-credential' push`.
- The multiprocessing pool uses `spawn`, so run entry points as modules (`python -m ...`), never `python - <<EOF`.
