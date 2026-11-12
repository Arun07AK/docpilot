"""Re-dump parsed sections and re-point gold headings after a parser change.

Each question's gold section is found again by its verbatim evidence within the
same doc, so labels never silently drift when parsing improves.
"""
import json
from pathlib import Path

from docpilot.ingest.parse import parse_markdown
from docpilot.ingest.pipeline import discover

HERE = Path(__file__).parent


def main():
    secs = [s for p, d, src in discover() for s in parse_markdown(p, d, src)]
    with open(HERE / "sections.jsonl", "w") as f:
        for s in secs:
            f.write(json.dumps({"doc_path": s.doc_path, "heading_path": s.heading_path, "text": s.text}) + "\n")
    qs = [json.loads(l) for l in (HERE / "questions.jsonl").read_text().splitlines() if l.strip()]
    changed = 0
    for q in qs:
        cands = [s for s in secs if s.doc_path == q["gold_doc"] and q["evidence"] in s.text]
        assert len(cands) == 1, (q["id"], len(cands))
        if cands[0].heading_path != q["gold_heading"]:
            print(f"{q['id']}: {q['gold_heading']!r} -> {cands[0].heading_path!r}")
            q["gold_heading"] = cands[0].heading_path
            changed += 1
    (HERE / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    print(f"{len(secs)} sections, {changed} labels re-pointed, all {len(qs)} evidence strings located")


if __name__ == "__main__":
    main()
