"""Markdown -> sections keyed by heading path.

The FastAPI and Pydantic docs are MkDocs Markdown. This parser does four things:
  1. strips front matter, HTML comments and heading anchors (`## Body { #body }`),
  2. inlines FastAPI's code includes (`{* ../../docs_src/x.py hl[3] *}`) from docs_src,
  3. unwraps admonitions (`/// tip` ... `///`, `!!! note`) so their prose stays,
  4. splits on #, ## and ### headings, never on a `#` inside a code fence.

Each section carries its full heading path ("Request Body > Import Pydantic's
BaseModel"). That path becomes the citation and the retrieval-eval label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*(\{\s*#[^}]*\})?\s*$")
_INCLUDE = re.compile(r"^\{\*\s*(\S+)(.*?)\*\}\s*$")
_SNIPPET = re.compile(r"^--8<--.*$")
_ADMON_OPEN = re.compile(r"^///\s*(\w+)?\s*(\|\s*(.*))?$")
_BANG_ADMON = re.compile(r'^(!!!|\?\?\?\+?)\s+(\w+)(\s+"(.*)")?\s*$')
_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HTML_TAG = re.compile(r"</?(?:abbr|span|div|img|a|br|details|summary)[^>]*>")


@dataclass(frozen=True)
class Section:
    doc_path: str  # e.g. "fastapi/tutorial/body.md"
    title: str  # document H1
    heading_path: str  # "Request Body > Create your data model"
    text: str  # body text of this section only


def _inline_include(target: str, md_file: Path, docs_src_root: Path | None) -> str:
    """Resolve `{* ../../docs_src/body/tutorial001_py310.py *}` to a code block."""
    if docs_src_root is None:
        return ""
    rel = target.split("docs_src/", 1)[-1] if "docs_src/" in target else None
    if rel is None:
        return ""
    path = docs_src_root / rel
    if not path.exists():
        return ""
    lang = "python" if path.suffix == ".py" else path.suffix.lstrip(".")
    return f"```{lang}\n{path.read_text(errors='ignore').rstrip()}\n```"


_BLOCK_MARKER = re.compile(r'^\s*(!!!|\?\?\?\+?|===)\s')
_ATTR = re.compile(r"\{\s*#[\w-]+\s*\}")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")


def _dedent_blocks(lines: list[str]) -> list[str]:
    """Pydantic nests admonition/tab bodies by 4-space indentation. Flatten them."""
    out, stack = [], []  # stack of required indents
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        if line.strip():
            while stack and indent < stack[-1]:
                stack.pop()
        cut = stack[-1] if stack else 0
        out.append(line[cut:] if line.strip() and indent >= cut else line.lstrip(" ") if not line.strip() else line)
        if _BLOCK_MARKER.match(line):
            stack.append(indent + 4)
    return out


def clean_lines(raw: str, md_file: Path, docs_src_root: Path | None) -> list[str]:
    raw = _FRONT_MATTER.sub("", raw)
    raw = _HTML_COMMENT.sub("", raw)
    out: list[str] = []
    in_fence = False
    for line in _dedent_blocks(raw.splitlines()):
        if not in_fence and line.lstrip().startswith(":::"):
            continue  # mkdocstrings API stub
        if not in_fence:
            m = re.match(r'^===\s+"(.*)"\s*$', line.strip())
            if m:
                out.append(f"{m.group(1)}:")
                continue
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line.strip() if line.strip().startswith(("```", "~~~")) else line)
            continue
        if in_fence:
            out.append(line)
            continue
        m = _INCLUDE.match(line.strip())
        if m:
            out.extend(_inline_include(m.group(1), md_file, docs_src_root).splitlines())
            continue
        if _SNIPPET.match(line.strip()):
            continue
        m = _ADMON_OPEN.match(line.strip())
        if m and line.strip().startswith("///"):
            label = (m.group(3) or m.group(1) or "").strip()
            if label:
                out.append(f"{label.capitalize()}:")
            continue
        m = _BANG_ADMON.match(line.strip())
        if m:
            label = (m.group(4) or m.group(2) or "").strip()
            out.append(f"{label.capitalize()}:")
            continue
        line = _HTML_TAG.sub("", line)
        line = _ATTR.sub("", line)
        line = _MD_LINK.sub(r"\1", line)
        line = _REF_LINK.sub(r"\1", line)
        out.append(line)
    return out


def parse_markdown(md_file: Path, doc_path: str, docs_src_root: Path | None = None) -> list[Section]:
    lines = clean_lines(md_file.read_text(errors="ignore"), md_file, docs_src_root)
    has_h1, fence = False, False
    for l in lines:  # a "# comment" inside a code fence is not a heading
        if _FENCE.match(l):
            fence = not fence
        elif not fence and l.startswith("# ") and _HEADING.match(l):
            has_h1 = True
            break
    # pages without an H1 get their title from the file name (MkDocs nav does the same)
    title = "" if has_h1 else (md_file.stem if md_file.stem != "index" else md_file.parent.name).replace("_", " ").replace("-", " ").capitalize()
    stack: list[str] = []  # heading stack, index = level - 1
    body: list[str] = []
    sections: list[Section] = []
    in_fence = False

    def flush():
        text = "\n".join(body).strip()
        if text and stack:
            sections.append(Section(doc_path, title or stack[0], " > ".join(stack), text))
        body.clear()

    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            name = _MD_LINK.sub(r"\1", m.group(2)).replace("`", "").strip()
            if level == 1 and not title:
                title = name
            if not has_h1:
                level = max(level, 2)  # page title stands in as the H1
                if not stack:
                    stack.append(title)
            del stack[level - 1:]
            while len(stack) < level - 1:  # skipped level, e.g. H2 -> H4
                stack.append(stack[-1] if stack else (title or name))
            stack.append(name)
            continue
        body.append(line)
    flush()
    if not sections and body:
        pass
    return sections
