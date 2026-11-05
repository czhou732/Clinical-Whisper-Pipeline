"""End-to-end analysis: transcripts + scores -> parameters -> validation report.

    # 1. List every interviewer prompt, for raters to label valence (blind to outcomes).
    uv run python -m research.speech_mechanisms.analysis list-prompts \\
        --transcripts DAIC/transcripts --source daic --out prompts_to_label.csv

    # 2. Run the pre-registered analysis.
    uv run python -m research.speech_mechanisms.analysis run \\
        --transcripts DAIC/transcripts --source daic --valence prompts_labeled.csv \\
        --scores DAIC/train_split.csv --out report.md

    # 3. Optional: agreement between human-timed and ClinicalWhisper transcripts.
    ... run ... --compare-transcripts CW/transcripts --compare-source cw

``--scores`` accepts DAIC-WOZ split files (``Participant_ID``, ``PHQ8_Score``,
``PHQ8_NoInterest``, ``PHQ8_Depressed``, ``PHQ8_Moving``) or a plain CSV with
``participant, item1, item2, item8, total``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from research.speech_mechanisms import turns
from research.speech_mechanisms.model_reactivity import (
    pairs_frame,
    participant_reactivity,
    reactivity_test,
)
from research.speech_mechanisms.model_timing import (
    CONTEXT_PARAMS,
    TIMING_PARAMS,
    participant_timing,
)
from research.speech_mechanisms.transcripts import load_dir, prompt_inventory
from research.speech_mechanisms.validate import agreement, compare_models, discriminant

_DAIC_COLUMNS = {"Participant_ID": "participant", "PHQ8_NoInterest": "item1",
                 "PHQ8_Depressed": "item2", "PHQ8_Moving": "item8", "PHQ8_Score": "total",
                 "Gender": "sex"}


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=_DAIC_COLUMNS)
    df["participant"] = df["participant"].astype(str)
    missing = {"item1", "item2", "item8", "total"} - set(df.columns)
    if missing:
        raise ValueError(f"scores file lacks {sorted(missing)} (item-level PHQ-8 needed)")
    keep = ["item1", "item2", "item8", "total"] + (["sex"] if "sex" in df else [])
    return df.set_index("participant")[keep]


def load_valence(path: Path) -> dict[str, str]:
    """Rater-labelled prompts: CSV ``prompt,valence`` (blank valence = unlabelled)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["prompt"]: r["valence"].strip().lower()
                for r in csv.DictReader(fh) if r.get("valence", "").strip()}


def parameters(transcripts: dict[str, list[dict]], valence_map: dict[str, str]):
    """``(timing, reactivity, pairs_frame)``, each indexed by participant.

    ``timing`` holds only Model A predictors; context measures such as answer
    length are in ``frame.attrs["context"]``.
    """
    pairs = {pid: turns.pairs(segs, valence_map) for pid, segs in transcripts.items()}
    everything = pd.DataFrame({pid: participant_timing(p) for pid, p in pairs.items()}).T
    frame = pairs_frame(pairs)
    frame.attrs["context"] = everything[CONTEXT_PARAMS].astype(float)
    reactivity = participant_reactivity(frame)
    return everything[TIMING_PARAMS].astype(float), reactivity, frame


def analyze(transcripts, valence_map, scores, compare=None) -> str:
    timing, reactivity, frame = parameters(transcripts, valence_map)
    params = timing.join(reactivity, how="outer").join(frame.attrs["context"], how="left")
    lines = [
        "# Speech mechanisms: timing vs reward reactivity", "",
        f"{len(transcripts)} interviews, {len(scores.index.intersection(params.index))} with "
        f"PHQ-8 items; {int(frame['positive'].sum())} positive-prompt responses of {len(frame)} "
        "labelled.", "",
        "## 1. Discriminant validity — standardized coefficients (items entered together)", "",
        "| parameter | n | item 1 (anhedonia) | item 2 (mood) | item 8 (psychomotor) | strongest |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in discriminant(params, scores).iterrows():
        cells = " | ".join(
            f"{r[f'{i}_beta']:+.2f} [{r[f'{i}_ci'][0]:+.2f}, {r[f'{i}_ci'][1]:+.2f}]"
            for i in ("item1", "item2", "item8")
        )
        lines.append(f"| {r['parameter']} | {r['n']} | {cells} | {r['strongest']} |")

    lines += ["", "## 2. Reward-reactivity test (pre-registered H2)", "",
              "`feature ~ positive * item1 + item2 + (1 | participant)`; key term `positive:item1`.", "",
              "| feature | positive × item1 | 95% CI | p |", "|---|---|---|---|"]
    for _, r in reactivity_test(frame, scores).iterrows():
        lines.append(f"| {r['feature']} | {r['interaction']:+.3f} | "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p']:.4f} |")

    lines += ["", "## 3. Which mechanism predicts symptoms? (nested CV, Spearman rho)", ""]
    sets = {"timing (A)": timing.dropna(axis=1, how="all"),
            "reactivity (B)": reactivity.dropna(axis=1, how="all")}
    sets["A + B"] = sets["timing (A)"].join(sets["reactivity (B)"], how="inner")
    for target in ("item8", "item1", "total"):
        res = compare_models(sets, scores[target])
        lines += [f"**Target: {target}** (n = {res['n']})", "",
                  "| features | mean rho | 95% of repeats |", "|---|---|---|"]
        for name, (mean, lo, hi) in res["rho"].items():
            lines.append(f"| {name} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        lines += ["", "| comparison | mean diff | corrected t | p |", "|---|---|---|---|"]
        for _, c in res["comparisons"].iterrows():
            lines.append(f"| {c['a']} vs {c['b']} | {c['mean_diff']:+.3f} | {c['t']:+.2f} | {c['p']:.4f} |")
        lines.append("")

    if compare is not None:
        c_timing, c_react, _ = parameters(compare, valence_map)
        lines += ["## 4. Measurement agreement: human-timed vs ClinicalWhisper", "",
                  "| parameter | n | ICC(2,1) |", "|---|---|---|"]
        for _, r in agreement(params, c_timing.join(c_react, how="outer")).iterrows():
            lines.append(f"| {r['parameter']} | {r['n']} | {r['icc_2_1']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    lp = sub.add_parser("list-prompts")
    lp.add_argument("--transcripts", type=Path, required=True)
    lp.add_argument("--source", choices=["daic", "cw"], required=True)
    lp.add_argument("--out", type=Path, required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--transcripts", type=Path, required=True)
    rn.add_argument("--source", choices=["daic", "cw"], required=True)
    rn.add_argument("--valence", type=Path, required=True)
    rn.add_argument("--scores", type=Path, required=True)
    rn.add_argument("--out", type=Path, required=True)
    rn.add_argument("--compare-transcripts", type=Path)
    rn.add_argument("--compare-source", choices=["daic", "cw"], default="cw")
    args = ap.parse_args()

    transcripts = load_dir(args.transcripts, args.source)
    if args.command == "list-prompts":
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt", "interviews", "valence"])
            for prompt, count in prompt_inventory(transcripts):
                w.writerow([prompt, count, ""])
        print(f"wrote {args.out}: label the valence column positive / neutral / negative")
        return
    compare = load_dir(args.compare_transcripts, args.compare_source) if args.compare_transcripts else None
    report = analyze(transcripts, load_valence(args.valence), load_scores(args.scores), compare)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
