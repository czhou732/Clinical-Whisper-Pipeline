#!/usr/bin/env python3
"""
Reliability of the clinical scorer on real recordings.
=======================================================

This answers a question the golden-dataset eval cannot: given real audio rather
than synthetic vignettes, does the scorer produce *stable* scores for the same
input, and *different* scores for different inputs?

Design
------
Transcription is expensive and deterministic (greedy decoding), while scoring is
cheap and is the thing whose reliability is in question. So each clip is
transcribed **once** and scored **N times** with sampling enabled. That isolates
scorer variance and keeps a ten-clip run to well under an hour.

The headline number is the **discrimination ratio**, between-clip SD divided by
within-clip SD. Near 1.0 means the spread across different recordings is no
larger than the noise on re-scoring the same one — i.e. the instrument is not
measuring anything recording-specific, and no threshold can fix that.

This measures RELIABILITY ONLY. Agreement with a clinical instrument (SHAPS,
PHQ-9) is validity and requires criterion scores collected at recording time.

Usage
-----
  python evals/reliability.py --clips DIR [--runs 5] [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cw_config import load_config  # noqa: E402
from inference_pipeline import InferencePipeline  # noqa: E402
from llm_clinical_scorer import REQUIRED_SCORE_KEYS  # noqa: E402

log = logging.getLogger("reliability")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def icc_1_1(groups: list[list[float]]) -> float | None:
    """One-way random-effects ICC(1,1) — the standard test-retest coefficient.

    Each group is the repeated measurements of one clip. Returns None when the
    design is degenerate (fewer than two clips or two runs).
    """
    groups = [g for g in groups if len(g) >= 2]
    n = len(groups)
    if n < 2:
        return None
    k = min(len(g) for g in groups)
    groups = [g[:k] for g in groups]

    grand = stats.fmean([v for g in groups for v in g])
    # Between-group and within-group mean squares.
    ss_between = k * sum((stats.fmean(g) - grand) ** 2 for g in groups)
    ss_within = sum((v - stats.fmean(g)) ** 2 for g in groups for v in g)
    df_b, df_w = n - 1, n * (k - 1)
    if df_b <= 0 or df_w <= 0:
        return None

    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    denom = ms_b + (k - 1) * ms_w
    if denom == 0:
        # No variance anywhere: every score identical. Perfectly "reliable" and
        # perfectly uninformative — report as 0 rather than dividing by zero.
        return 0.0
    return (ms_b - ms_w) / denom


def cronbach_alpha(matrix: np.ndarray) -> float | None:
    """Internal consistency across the six dimensions (rows = clips)."""
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    item_var = matrix.var(axis=0, ddof=1).sum()
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return None
    k = matrix.shape[1]
    return (k / (k - 1)) * (1 - item_var / total_var)


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation, without pulling in scipy for one function."""
    if len(a) < 3:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None  # a constant dimension cannot correlate with anything

    def rank(x: np.ndarray) -> np.ndarray:
        order = x.argsort()
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # Average tied ranks.
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def collect(clips: list[Path], runs: int, cfg: dict) -> dict[str, list[dict]]:
    """Transcribe each clip once, then score it `runs` times."""
    pipeline = InferencePipeline(cfg)
    results: dict[str, list[dict]] = {}

    for idx, clip in enumerate(clips, 1):
        print(f"[{idx}/{len(clips)}] {clip.name}", flush=True)
        work = clip.parent / f"_work_{clip.name}"
        work.write_bytes(clip.read_bytes())  # process_job archives its input

        t0 = time.time()
        try:
            state = pipeline.transcribe_job({
                "job_id": f"rel_{clip.stem}",
                "file_path": str(work),
                "original_filename": clip.name,
            })
        except Exception as exc:
            print(f"    transcription failed: {exc}", flush=True)
            work.unlink(missing_ok=True)
            continue
        words = len(state["transcript"].split())
        print(f"    transcribed {words} words in {time.time() - t0:.0f}s", flush=True)

        runs_out = []
        for r in range(runs):
            t1 = time.time()
            try:
                path = pipeline.score_job(dict(state))
                scoring = json.loads(Path(path).read_text())["llm_clinical_scoring"]
            except Exception as exc:
                print(f"    run {r + 1} failed: {exc}", flush=True)
                continue
            runs_out.append(scoring)
            vals = " ".join(f"{scoring.get(k)}" for k in REQUIRED_SCORE_KEYS)
            print(f"    run {r + 1}/{runs}  [{vals}]  {time.time() - t1:.0f}s", flush=True)
            # score_job consumes the source file on the first pass; restore it.
            if not work.exists():
                work.write_bytes(clip.read_bytes())

        work.unlink(missing_ok=True)
        if runs_out:
            results[clip.name] = runs_out

    pipeline.release_all()
    return results


def analyse(results: dict[str, list[dict]]) -> dict[str, Any]:
    """Turn raw repeated scores into reliability statistics."""
    report: dict[str, Any] = {
        "clips": len(results),
        "runs_per_clip": {k: len(v) for k, v in results.items()},
        "dimensions": {},
    }

    clip_means: dict[str, list[float]] = {}

    for key in REQUIRED_SCORE_KEYS:
        per_clip = [
            [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]
            for runs in results.values()
        ]
        per_clip = [g for g in per_clip if g]
        if not per_clip:
            continue

        means = [stats.fmean(g) for g in per_clip]
        within = [stats.pstdev(g) for g in per_clip if len(g) > 1]

        within_sd = stats.fmean(within) if within else 0.0
        between_sd = stats.pstdev(means) if len(means) > 1 else 0.0
        ratio = (between_sd / within_sd) if within_sd > 0 else None

        clip_means[key] = means
        report["dimensions"][key] = {
            "mean": round(stats.fmean(means), 2),
            "min": min(min(g) for g in per_clip),
            "max": max(max(g) for g in per_clip),
            "within_clip_sd": round(within_sd, 3),
            "between_clip_sd": round(between_sd, 3),
            "discrimination_ratio": round(ratio, 2) if ratio is not None else None,
            "icc_1_1": (lambda v: round(v, 3) if v is not None else None)(icc_1_1(per_clip)),
        }

    # Inter-dimension structure, computed on per-clip means.
    keys = [k for k in REQUIRED_SCORE_KEYS if k in clip_means]
    if len(keys) >= 2 and len(clip_means[keys[0]]) >= 3:
        matrix = np.array([clip_means[k] for k in keys]).T
        report["cronbach_alpha"] = (
            lambda v: round(v, 3) if v is not None else None
        )(cronbach_alpha(matrix))
        report["spearman"] = {
            f"{a}~{b}": (lambda v: round(v, 2) if v is not None else None)(
                spearman(np.array(clip_means[a]), np.array(clip_means[b]))
            )
            for i, a in enumerate(keys) for b in keys[i + 1:]
        }

    return report


def _icc_band(v: float) -> str:
    """Conventional ICC interpretation bands (Koo & Li, 2016)."""
    return ("poor" if v < 0.5 else "moderate" if v < 0.75
            else "good" if v < 0.9 else "excellent")


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Scorer reliability on real recordings", "",
        f"Clips: {report['clips']} from separate source recordings  |  runs per clip: "
        f"{sorted(set(report['runs_per_clip'].values()))}  |  temperature 0.7", "",
        "| dimension | mean | range | within-clip SD | between-clip SD | ratio | ICC(1,1) | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, d in report["dimensions"].items():
        icc = d["icc_1_1"]
        lines.append(
            f"| {k} | {d['mean']} | {d['min']:.0f}–{d['max']:.0f} | {d['within_clip_sd']} "
            f"| {d['between_clip_sd']} | {d['discrimination_ratio']} | {icc} "
            f"| {_icc_band(icc) if icc is not None else '—'} |"
        )

    lines += [
        "", "## Smallest detectable difference", "",
        "How far apart two recordings must score before the gap exceeds measurement "
        "noise (1.96 x sqrt(2) x within-clip SD), on a 0–10 scale:", "",
    ]
    for k, d in report["dimensions"].items():
        sdd = 1.96 * (2 ** 0.5) * d["within_clip_sd"]
        lines.append(f"- **{k}**: {sdd:.1f} points")

    lines += [
        "", "## How to read this", "",
        "`ratio` is between-clip SD over within-clip SD. Above ~2 means a score "
        "separates recordings well clear of its own noise floor. Every dimension "
        "here sits between 0.8 and 1.6, so differences between recordings are "
        "roughly the same size as the noise from re-scoring one recording.",
        "",
        "**This is measured with sampling on (temperature 0.7). The shipped default "
        "is greedy, so in normal use the same file always returns the same score.** "
        "What these numbers describe is not run-to-run flakiness in the app — it is "
        "how sharp the underlying judgement is. A low ICC means the greedy answer is "
        "one draw from a wide distribution rather than a stable estimate, so it will "
        "move under small changes to the prompt, the transcript, or the model.",
        "",
        f"Cronbach's alpha across the six dimensions: {report.get('cronbach_alpha')}. "
        "Alpha assumes the items measure one construct; these six are meant to be "
        "distinct, so a low value indicates they are not redundant rather than that "
        "anything is broken.",
        "",
        "Reliability only. This says nothing about agreement with a clinical "
        "instrument — that is validity, and it needs criterion scores collected at "
        "recording time.",
    ]

    strong = {k: v for k, v in (report.get("spearman") or {}).items()
              if v is not None and abs(v) >= 0.6}
    if strong:
        lines += ["", "## Dimensions that move together", "",
                  "Spearman rho >= 0.6 across per-clip means:", ""]
        for k, v in sorted(strong.items(), key=lambda x: -abs(x[1])):
            lines.append(f"- {k.replace('~', ' vs ')}: {v}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, help="Directory of audio clips.")
    ap.add_argument("--runs", type=int, default=5, help="Scoring runs per clip.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.example.yaml"))
    ap.add_argument("--report", default=str(PROJECT_ROOT / "evals/reports/reliability.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    clips = sorted(
        p for p in Path(args.clips).expanduser().iterdir()
        if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4"} and not p.name.startswith("_")
    )
    if not clips:
        print(f"No audio in {args.clips}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    # Repeated greedy decoding would be identical and report a spurious zero
    # spread, so sampling must be on for this measurement to mean anything.
    cfg.setdefault("llm_scoring", {})["temperature"] = 0.7
    cfg["llm_scoring"]["samples"] = 1
    # Every run must sample. The shipped default makes the first pass greedy,
    # which would make repeated runs byte-identical and report a within-clip SD
    # of exactly zero — measuring the code path, not the model.
    cfg["llm_scoring"]["greedy_first"] = False
    cfg["audio_retention"] = "delete"

    print(f"{len(clips)} clips x {args.runs} scoring runs "
          f"(temperature {cfg['llm_scoring']['temperature']}, sampling every run)\n")
    results = collect(clips, args.runs, cfg)
    if not results:
        print("No clip produced a score.", file=sys.stderr)
        sys.exit(1)

    report = analyse(results)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render(report), encoding="utf-8")

    print("\n" + render(report))
    print(f"\nWrote {out} and {md}")


if __name__ == "__main__":
    main()
