"""Fetch the corpus: FastAPI and Pydantic docs at pinned git tags.

Sparse, shallow clones keep the download small. The commit SHAs are written to
data/SOURCES.md, so every number in the README can be traced to an exact corpus.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from docpilot.config import settings


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    tag: str
    sparse: tuple[str, ...]  # paths to check out
    docs_root: str  # markdown root inside the checkout


SOURCES = (
    Source("fastapi", "https://github.com/fastapi/fastapi.git", "0.162.5",
           ("docs/en/docs", "docs_src"), "docs/en/docs"),
    Source("pydantic", "https://github.com/pydantic/pydantic.git", "v2.14.3",
           ("docs",), "docs"),
)


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch(dest: Path | None = None) -> dict[str, Path]:
    dest = dest or settings.corpus_dir
    dest.mkdir(parents=True, exist_ok=True)
    roots: dict[str, Path] = {}
    lines = ["# Corpus sources", "", "| source | tag | commit |", "|---|---|---|"]
    for s in SOURCES:
        target = dest / s.name
        if not (target / ".git").exists():
            _git("clone", "--depth", "1", "--branch", s.tag, "--filter=blob:none",
                 "--sparse", s.repo, str(target))
            _git("sparse-checkout", "set", *s.sparse, cwd=target)
        sha = _git("rev-parse", "HEAD", cwd=target)
        lines.append(f"| {s.name} | {s.tag} | `{sha}` |")
        roots[s.name] = target / s.docs_root
    (dest.parent / "SOURCES.md").write_text("\n".join(lines) + "\n")
    return roots


if __name__ == "__main__":
    for name, root in fetch().items():
        print(name, root, sum(1 for _ in root.rglob("*.md")), "markdown files")
