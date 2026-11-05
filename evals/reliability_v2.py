#!/usr/bin/env python3
"""Reliability of clinical scorer version 2 (four 0-3 scores with quoted evidence).

Each transcript is scored N times with sampling on (temperature 0.7, every
run sampled), and ICC(1,1) is computed per score exactly as for version 1
(one-way random effects, Shrout & Fleiss 1979; see reliability.py). A run that
found no evidence for a score leaves it missing; transcripts with a missing
run are left out of that score's ICC, and the count is reported.

Inputs: the 12 synthetic vignettes in clinical_golden_dataset.jsonl, plus any
extra transcripts given with --transcripts (analysis JSON files). This
measures reliability (does the scorer agree with itself), and, on the
vignettes, whether scores move in the expected direction between groups. It
does not measure accuracy against a clinical standard; that needs PHQ or
SCID-rated recordings.

Usage:  python evals/reliability_v2.py --runs 5 --out evals/reports/reliability_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from reliability import anova_components  # noqa: E402

import llm_clinical_scorer as scorer  # noqa: E402
from cw_config import load_config  # noqa: E402
from transcript_formatter import format_structured_transcript  # noqa: E402


def vignettes() -> list[tuple[str, str, str]]:
    out = []
    for line in (ROOT / "evals" / "clinical_golden_dataset.jsonl").read_text().splitlines():
        row = json.loads(line)
        speakers = []
        for seg in row["segments"]:
            if seg["speaker"] not in speakers:
                speakers.append(seg["speaker"])
        roles = {speakers[0]: "Interviewer", **({speakers[1]: "Subject"} if len(speakers) > 1 else {})}
        out.append((row["id"], row["category"], format_structured_transcript(row["segments"], roles)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--transcripts", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(None)
    cfg.setdefault("llm_scoring", {}).update(samples=1, temperature=0.7, greedy_first=False)
    items = vignettes()
    for path in args.transcripts:
        d = json.loads(Path(path).read_text())
        items.append((Path(path).stem, "recording", d["structured_transcript"]))

    runs: dict[str, list[dict]] = {}
    for name, category, text in items:
        runs[name] = []
        for i in range(args.runs):
            r = scorer.score_transcript(text, config=cfg)
            runs[name].append({k: r.get(k) for k in scorer.REQUIRED_SCORE_KEYS})
            print(f"{name} run {i + 1}: {runs[name][-1]}", flush=True)

    report: dict = {"scorer_version": scorer.SCORER_VERSION, "runs": args.runs,
                    "transcripts": len(items), "per_score": {}, "raw": runs,
                    "categories": {n: c for n, c, _ in items}}
    for key in scorer.REQUIRED_SCORE_KEYS:
        groups = [[r[key] for r in rs] for rs in runs.values()]
        complete = [g for g in groups if all(isinstance(v, (int, float)) for v in g)]
        comp = anova_components(complete) if len(complete) >= 3 else None
        missing_runs = sum(1 for g in groups for v in g if v is None)
        report["per_score"][key] = {
            "transcripts_used": len(complete), "missing_runs": missing_runs,
            **({k: comp[k] for k in ("icc_1_1", "icc_ci95") if k in comp} if comp else {}),
        }
        # Direction check on the vignettes: group means by category.
        by_cat: dict[str, list[float]] = {}
        for (name, cat, _), g in zip(items, groups):
            vals = [v for v in g if isinstance(v, (int, float))]
            if vals:
                by_cat.setdefault(cat, []).append(sum(vals) / len(vals))
        report["per_score"][key]["category_means"] = {c: round(sum(v) / len(v), 2) for c, v in by_cat.items()}
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["per_score"], indent=1))


if __name__ == "__main__":
    main()
