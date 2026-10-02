#!/usr/bin/env python3
"""Turn annotated real transcripts into a masking gold set, and check agreement.

For the real-transcript gold set. It runs on the iLab machine where the
consented recordings live; the annotated files hold identifiers and never
leave it. Only evaluate.py's summary numbers (no text) are shared.

Annotation: each annotator copies a raw transcript (one segment per line,
"SPEAKER: text") and wraps every identifier in double braces with its type:

    S02: I moved in with my sister {{first_name:Elena}} in {{city:Fresno}}.

Types: first_name, last_name, full_name, city, street, organization, date,
age, phone, email, id_number, url, other. Mark every person other than
public figures, every place smaller than a state, every date more specific
than a year, and anything else that could identify someone (HIPAA Safe
Harbor's 18 identifiers). When unsure, mark it.

    # Agreement between two annotators of the same transcripts
    python evals/masking/gold_from_markup.py agree A/ B/
    # Adjudicated files -> gold set for evaluate.py
    python evals/masking/gold_from_markup.py build ADJUDICATED/ --out gold.jsonl
    python evals/masking/evaluate.py --set /abs/path/gold.jsonl --whole
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MARK = re.compile(r"\{\{([a-z_]+):(.+?)\}\}")
TYPES = {"first_name", "last_name", "full_name", "city", "street", "organization", "date", "age",
         "phone", "email", "id_number", "url", "other"}


def parse_line(line: str) -> tuple[str, str, list[dict]]:
    """("S02", plain text, [{start, end, type}]) for one annotated line."""
    speaker, _, body = line.partition(":") if ":" in line[:20] else ("", "", line)
    text, ents, last = "", [], 0
    for m in MARK.finditer(body):
        if m.group(1) not in TYPES:
            raise ValueError(f"unknown type {m.group(1)!r} in: {line.strip()}")
        text += body[last:m.start()]
        ents.append({"start": len(text), "end": len(text) + len(m.group(2)), "type": m.group(1),
                     "origin": "real"})
        text += m.group(2)
        last = m.end()
    text += body[last:]
    lead = len(text) - len(text.lstrip())
    for e in ents:
        e["start"] -= lead
        e["end"] -= lead
    return speaker.strip(), text.strip(), ents


def read(folder: Path) -> list[dict]:
    rows = []
    for f in sorted(folder.glob("*.txt")):
        for i, line in enumerate(f.read_text().splitlines()):
            if line.strip():
                speaker, text, ents = parse_line(line)
                rows.append({"id": f"{f.stem}:{i:04d}", "doc": f.stem, "speaker": speaker,
                             "text": text, "entities": ents})
    return rows


def agreement(a: list[dict], b: list[dict]) -> dict:
    """Exact-span F1 between two annotators, and overlap F1 (any shared character)."""
    by_id = {r["id"]: r for r in b}
    tp_exact = tp_overlap = n_a = n_b = 0
    for r in a:
        other = by_id.get(r["id"])
        if other is None or other["text"] != r["text"]:
            raise ValueError(f"{r['id']}: the two annotators' text differs; compare the same raw transcripts")
        sa = {(e["start"], e["end"]) for e in r["entities"]}
        sb = {(e["start"], e["end"]) for e in other["entities"]}
        n_a, n_b = n_a + len(sa), n_b + len(sb)
        tp_exact += len(sa & sb)
        tp_overlap += sum(any(s < e2 and s2 < e for s2, e2 in sb) for s, e in sa)
    f1 = lambda tp: round(2 * tp / (n_a + n_b), 3) if n_a + n_b else None  # noqa: E731
    return {"annotator_a": n_a, "annotator_b": n_b, "exact_f1": f1(tp_exact), "overlap_f1": f1(tp_overlap)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree")
    g.add_argument("a", type=Path)
    g.add_argument("b", type=Path)
    g = sub.add_parser("build")
    g.add_argument("folder", type=Path)
    g.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "agree":
        print(json.dumps(agreement(read(args.a), read(args.b)), indent=1))
    else:
        rows = read(args.folder)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} segments, {sum(len(r['entities']) for r in rows)} identifiers, "
              f"{len({r['doc'] for r in rows})} transcripts -> {args.out}")


if __name__ == "__main__":
    main()
