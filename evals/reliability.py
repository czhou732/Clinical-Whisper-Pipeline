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
import math
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

def anova_components(groups: list[list[float]]) -> dict[str, Any] | None:
    """One-way random-effects ANOVA on repeated measurements of each target.

    Returns ICC(1,1) with a 95% confidence interval, plus the variance
    components the descriptive statistics should be derived from.

    ICC(1,1) is the correct form here (Shrout & Fleiss 1979): the repeated
    scores are exchangeable draws from one stochastic process, not a fixed panel
    of identifiable raters, so there is no rater factor to model. Run 2 of clip A
    bears no relationship to run 2 of clip B, which is exactly the one-way case.

    Verified against the Shrout & Fleiss (1979) Table 1 worked example
    (published ICC(1,1) = 0.17; this returns 0.1657).
    """
    groups = [g for g in groups if len(g) >= 2]
    n = len(groups)
    if n < 2:
        return None
    k = min(len(g) for g in groups)
    groups = [g[:k] for g in groups]

    grand = stats.fmean([v for g in groups for v in g])
    df_b, df_w = n - 1, n * (k - 1)
    if df_b <= 0 or df_w <= 0:
        return None

    ms_b = k * sum((stats.fmean(g) - grand) ** 2 for g in groups) / df_b
    ms_w = sum((v - stats.fmean(g)) ** 2 for g in groups for v in g) / df_w

    denom = ms_b + (k - 1) * ms_w
    icc = 0.0 if denom == 0 else (ms_b - ms_w) / denom

    # SEM is the pooled within-subject SD, sqrt(MSW) — not the mean of the
    # per-target SDs, which underestimates it.
    sem = math.sqrt(ms_w)
    # Between-subject SD as a variance component, rather than the SD of the
    # observed means (which carries measurement error).
    sd_between = math.sqrt(max(0.0, (ms_b - ms_w) / k))

    ci_low = ci_high = None
    if ms_w > 0:
        try:
            from scipy import stats as sps
            f_obs = ms_b / ms_w
            f_l = f_obs / sps.f.ppf(0.975, df_b, df_w)
            f_u = f_obs * sps.f.ppf(0.975, df_w, df_b)
            ci_low = (f_l - 1) / (f_l + k - 1)
            ci_high = (f_u - 1) / (f_u + k - 1)
        except Exception:  # pragma: no cover - CI is a nicety, not the result
            pass

    return {
        "icc_1_1": round(icc, 3),
        "icc_ci95": [round(ci_low, 3), round(ci_high, 3)]
        if ci_low is not None else None,
        "sem": round(sem, 3),
        "sd_between": round(sd_between, 3),
        "n_targets": n,
        "k_runs": k,
    }


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
        clip_means[key] = means

        comp = anova_components(per_clip)
        if comp is None:
            continue

        sem, sd_b = comp["sem"], comp["sd_between"]
        report["dimensions"][key] = {
            "mean": round(stats.fmean(means), 2),
            "min": min(min(g) for g in per_clip),
            "max": max(max(g) for g in per_clip),
            # SEM = sqrt(MSW), the pooled within-target SD.
            "sem": sem,
            "sd_between": sd_b,
            "discrimination_ratio": round(sd_b / sem, 2) if sem > 0 else None,
            # MDC95 = 1.96 * sqrt(2) * SEM — the smallest change that exceeds
            # measurement error at 95% confidence.
            "mdc_95": round(1.96 * math.sqrt(2) * sem, 2),
            "icc_1_1": comp["icc_1_1"],
            "icc_ci95": comp["icc_ci95"],
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


def _average_measures(icc: float, k: int) -> float:
    """ICC(1,k) from ICC(1,1): reliability of the mean of k exchangeable runs."""
    return k * icc / (1 + (k - 1) * icc)


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Scorer reliability on real recordings", "",
        f"{report['clips']} clips from separate source recordings, scored "
        f"{sorted(set(report['runs_per_clip'].values()))[0]} times each with "
        "sampling on (temperature 0.7).", "",
        "One-way random-effects ICC(1,1), per Shrout & Fleiss (1979): the repeated "
        "scores are exchangeable draws from one stochastic process, not a fixed "
        "panel of identifiable raters.", "",
        "| dimension | mean | range | SEM | SD between | ratio | ICC(1,1) | 95% CI | |"
        " ICC(1,k): mean of k runs | 95% CI | |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    runs = sorted(set(report["runs_per_clip"].values()))[0]
    for k, d in report["dimensions"].items():
        icc, ci = d["icc_1_1"], d.get("icc_ci95")
        ci_s = f"{ci[0]:.2f} – {ci[1]:.2f}" if ci else "—"
        # Average-measures ICC from the same one-way ANOVA: an exact function of
        # ICC(1,1) at this k — the reliability of llm_scoring.samples = k.
        icck = _average_measures(icc, runs)
        cik = (f"{_average_measures(max(ci[0], -0.24), runs):.2f} – "
               f"{_average_measures(ci[1], runs):.2f}") if ci else "—"
        lines.append(
            f"| {k} | {d['mean']} | {d['min']:.0f}–{d['max']:.0f} | {d['sem']} "
            f"| {d['sd_between']} | {d['discrimination_ratio']} | {icc} | {ci_s} "
            f"| {_icc_band(icc) if icc is not None else '—'} "
            f"| {icck:.2f} | {cik} | {_icc_band(icck)} |"
        )

    lines += [
        "", "## Definitions", "",
        "- **SEM** — standard error of measurement, sqrt(MSW) from the one-way "
        "ANOVA. The pooled within-target SD.",
        "- **SD between** — between-target variance component, "
        "sqrt((MSB - MSW) / k), which excludes measurement error.",
        "- **ratio** — SD between / SEM. Above ~2 means a score separates "
        "recordings well clear of its own noise floor.",
        "- **MDC95** — smallest detectable change, 1.96 * sqrt(2) * SEM: the gap "
        "two recordings must show before it exceeds measurement error.", "",
        "| dimension | MDC95 (0–10 scale) |", "|---|---|",
    ]
    for k, d in report["dimensions"].items():
        lines.append(f"| {k} | {d['mdc_95']} |")

    lines += [
        "", "## Limitations", "",
        f"- **n = {report['clips']} targets** is well below the ~30 "
        "usually recommended for an ICC study, which is why the confidence "
        "intervals above are very wide. Treat the point estimates as indicative.",
        "- Clips were drawn two per source recording, so they are **clustered** "
        "rather than fully independent; the between-target component is likely "
        "overstated.",
        "- Measured with sampling on. The shipped default is greedy decoding, so "
        "in normal use the same file returns the same score. A low ICC does not "
        "mean the app is unstable — it means the single score is one draw from a "
        "wide distribution, and will move under small changes to prompt, "
        "transcript or model version.", "",
        f"Cronbach's alpha across the six dimensions: {report.get('cronbach_alpha')}. "
        "Alpha assumes the items measure one construct; these six are meant to be "
        "distinct, so a low value indicates they are not redundant rather than "
        "that anything is wrong.", "",
        "Reliability only. This says nothing about agreement with a clinical "
        "instrument — that is validity, and needs criterion scores collected at "
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
    ap.add_argument("--samples", type=int, default=1,
                    help="Samples averaged within each run (the app's llm_scoring.samples).")
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
    # Each "run" is one scoring; --samples > 1 makes each run itself a mean of
    # that many samples, measuring the reliability of the shipped default.
    cfg["llm_scoring"]["samples"] = args.samples
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
