#!/usr/bin/env python3
"""Masking recall and precision on evals/masking/set.jsonl.

An identifier counts as caught when every character of it ends up inside a
tag (any tag type). Over-masking is counted as ordinary words lost to tags.
Run each sentence through ClinicalWhisper's own masker (PIIScrubber), so any
safety net added there is measured too.

    python evals/masking/evaluate.py [--threshold 0.7] [--label NAME]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAG = re.compile(r"^\[[a-z_]+(?:_\d+)?\][.,!?;:']*$|^\[REDACTED\]")


def masked_char_spans(raw: str, masked: str) -> list[tuple[int, int]]:
    """Character spans of ``raw`` whose words became tags in ``masked``."""
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", raw)]
    a = [re.sub(r"[^\w']", "", w.lower()) for _, _, w in words]
    b_words = masked.split()
    b = [re.sub(r"[^\w']", "", w.lower()) for w in b_words]
    spans = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "delete" or (op == "replace" and any(TAG.match(w) for w in b_words[j1:j2])):
            spans.extend((words[i][0], words[i][1]) for i in range(i1, i2))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-safety-net", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--set", default="set.jsonl", help="set.jsonl (tuning) or held_out.jsonl")
    ap.add_argument("--show-misses", action="store_true",
                    help="print every missed identifier in its sentence (screen only, never --out)")
    ap.add_argument("--whole", action="store_true",
                    help="mask each transcript ('doc') as one recording, as the app does")
    ap.add_argument("--masked", type=Path, default=None,
                    help="score another tool's output instead ({id, masked} per line)")
    args = ap.parse_args()

    set_path = Path(args.set) if Path(args.set).is_absolute() else Path(__file__).with_name(args.set)
    rows = [json.loads(l) for l in set_path.read_text().splitlines()]
    if args.masked:
        given = {d["id"]: d["masked"] for d in map(json.loads, args.masked.read_text().splitlines())}
    else:
        from pii_scrubber import PIIScrubber
        scrubber = PIIScrubber(confidence_threshold=args.threshold, strict=True,
                               safety_net=not args.no_safety_net)
    hit = defaultdict(lambda: [0, 0])          # by type: [caught, total]
    by_origin = defaultdict(lambda: [0, 0])    # names only
    words_total = words_lost = 0
    misses = []
    if args.whole and not args.masked:
        # Whole recordings: numbering and name propagation work across segments.
        given = {}
        for doc in dict.fromkeys(r.get("doc", r["id"]) for r in rows):
            seg = [r for r in rows if r.get("doc", r["id"]) == doc]
            out = scrubber.scrub_segments([{"text": r["text"]} for r in seg])
            given.update({r["id"]: o["text"] for r, o in zip(seg, out)})
    for r in rows:
        if args.masked or args.whole:
            masked = given[r["id"]]
        else:
            scrubber._ids = {}
            masked = scrubber.scrub_text(r["text"])
        spans = masked_char_spans(r["text"], masked)
        ent_chars = set()
        for e in r["entities"]:
            ent_chars.update(range(e["start"], e["end"]))
            ok = all(any(s <= c < e2 for s, e2 in spans) for c in range(e["start"], e["end"])
                     if not r["text"][c].isspace())
            hit[e["type"]][0] += ok
            hit[e["type"]][1] += 1
            if e["type"] in ("first_name", "full_name"):
                by_origin[e.get("origin", "unknown")][0] += ok
                by_origin[e.get("origin", "unknown")][1] += 1
            if not ok:
                misses.append({"id": r["id"], "type": e["type"], "text": r["text"][e["start"]:e["end"]],
                               "sentence": r["text"]})
        for m in re.finditer(r"\S+", r["text"]):
            if not any(m.start() <= c < m.end() for c in ent_chars):
                words_total += 1
                if any(s <= m.start() < e2 for s, e2 in spans):
                    words_lost += 1

    total_c = sum(v[0] for v in hit.values())
    total_n = sum(v[1] for v in hit.values())
    report = {
        "label": args.label, "threshold": None if args.masked else args.threshold, "sentences": len(rows),
        "recall": round(total_c / total_n, 3), "identifiers": total_n,
        "over_masking": round(words_lost / max(words_total, 1), 3),
        "recall_by_type": {k: round(v[0] / v[1], 3) for k, v in sorted(hit.items())},
        "name_recall_by_origin": {k: round(v[0] / v[1], 3) for k, v in sorted(by_origin.items())},
        # Quotes sentences: kept out of real-transcript (--whole) reports, whose
        # summary is meant to leave the machine.
        **({} if args.whole else {"misses_sample": misses[:25]}),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "misses_sample"}, indent=1))
    if args.show_misses:
        for m in misses:
            print(f"MISSED {m['type']}: {m['text']!r} in {m['sentence']!r}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
