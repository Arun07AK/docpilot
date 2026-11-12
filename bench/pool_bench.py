"""When does the multiprocessing pool pay off for parse + chunk?

The real corpus (179 pages) parses in ~0.3 s, less than the cost of spawning
workers, so the pool is slower at 1x. This replicates the corpus 1x, 4x and 16x
(with renamed files so the chunks stay distinct) and times serial vs pool.
  python -m bench.pool_bench
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from docpilot.config import settings
from docpilot.ingest.pipeline import chunk_corpus

OUT = Path(__file__).parent / "results"


def replicate(k: int) -> Path:
    root = Path(tempfile.mkdtemp())
    for name in ("fastapi", "pydantic"):
        src = settings.corpus_dir / name
        for i in range(k):
            dst_docs = root / name / ("docs/en/docs" if name == "fastapi" else "docs") / f"copy{i}"
            shutil.copytree(src / ("docs/en/docs" if name == "fastapi" else "docs"), dst_docs,
                            ignore=shutil.ignore_patterns("*.png", "*.jpg", "*.svg", "*.gif", "*.ico"))
            for p in dst_docs.rglob("*.md"):  # make every copy's text unique so nothing dedups
                p.write_text(p.read_text(errors="ignore").replace("\n\n", f"\n\n<!-- c{i} -->\n", 1) + f"\n\ncopy {i}\n")
        if name == "fastapi" and (src / "docs_src").exists():
            os.symlink(src / "docs_src", root / name / "docs_src")
    return root


def timed(fn, runs=3):
    xs = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        xs.append(time.perf_counter() - t)
    return statistics.median(xs), out


def main():
    rows = []
    cpus = os.cpu_count()
    for k in (1, 4, 16):
        root = replicate(k)
        try:
            serial, (ch, st) = timed(lambda: chunk_corpus(256, 32, workers=1, corpus_dir=root))
            pool, (ch2, _) = timed(lambda: chunk_corpus(256, 32, workers=cpus, corpus_dir=root))
            assert len(ch) == len(ch2)
            rows.append({"corpus_x": k, "files": st.files, "chunks": len(ch), "workers": cpus,
                         "serial_s": round(serial, 3), "pool_s": round(pool, 3), "speedup": round(serial / pool, 2)})
            print(rows[-1])
        finally:
            shutil.rmtree(root, ignore_errors=True)
    (OUT / "pool.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
