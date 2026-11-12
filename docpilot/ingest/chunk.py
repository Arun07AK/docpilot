"""Section-aware chunking with overlap.

Rules, in priority order:
  * a chunk never crosses a section boundary (citations stay exact);
  * paragraphs and code blocks are packed whole until the token budget is hit;
  * a single block larger than the budget is split on token windows;
  * the next chunk starts with the last `overlap` tokens of the previous one,
    so a sentence cut at a boundary is still retrievable from either side.

Every chunk's text is prefixed with its heading path. The embedding then "knows"
it is about, say, "Request Body > Create your data model", even when the body
never repeats those words.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from docpilot.ingest.parse import Section

_enc = None


def encoder():
    """Lazily created per process (each pool worker builds its own)."""
    global _enc
    if _enc is None:
        import tiktoken

        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


@dataclass(frozen=True)
class Chunk:
    id: str  # first 16 hex of sha256(text): stable, content-addressed
    doc_path: str
    title: str
    heading_path: str
    chunk_index: int
    text: str
    token_count: int
    content_hash: str

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def content_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalised.encode()).hexdigest()


def _blocks(text: str) -> list[str]:
    """Split into paragraphs, keeping each fenced code block as one unit."""
    blocks, cur, in_fence = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def chunk_section(sec: Section, max_tokens: int, overlap: int) -> list[Chunk]:
    enc = encoder()
    prefix = f"{sec.title} > {sec.heading_path}" if not sec.heading_path.startswith(sec.title) else sec.heading_path
    budget = max(32, max_tokens - len(enc.encode(prefix)) - 2)

    # token-level pieces: whole blocks when they fit, token windows when they don't
    pieces: list[list[int]] = []
    for block in _blocks(sec.text):
        toks = enc.encode(block)
        if len(toks) <= budget:
            pieces.append(toks)
        else:
            step = budget - overlap
            for i in range(0, len(toks), step):
                pieces.append(toks[i:i + budget])
                if i + budget >= len(toks):
                    break

    sep = enc.encode("\n\n")
    windows: list[list[int]] = []
    cur: list[int] = []
    for p in pieces:
        add = (sep if cur else []) + p
        if cur and len(cur) + len(add) > budget:
            windows.append(cur)
            tail = cur[-overlap:] if overlap else []
            cur = tail + sep + p if tail else list(p)
            if len(cur) > budget:  # overlap + oversized piece: drop the overlap
                cur = list(p)
        else:
            cur = cur + add
    if cur:
        windows.append(cur)

    out = []
    for i, w in enumerate(windows):
        text = f"{prefix}\n\n{enc.decode(w).strip()}"
        h = content_hash(text)
        out.append(Chunk(h[:16], sec.doc_path, sec.title, sec.heading_path, i, text,
                         len(w), h))
    return out
