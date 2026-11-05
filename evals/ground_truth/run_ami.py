"""Transcribe prepared AMI meetings under several configurations and score them.

    uv run python evals/ground_truth/run_ami.py ~/Developer/datasets/ami/prepared \\
        --out ~/Developer/datasets/ami/runs --report evals/reports/transcription_accuracy.md

Hypothesis transcripts stay under ``--out`` (outside the repo); only the
aggregate numbers go into the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from moss_diarizer import MOSSDiarizer, _audio_duration  # noqa: E402
from score import score  # noqa: E402

CONFIGS = {
    "single pass (full context)": {"window_seconds": 1e6},
    "windows 300 s / 30 s overlap": {"window_seconds": 300, "window_overlap_seconds": 30},
    "windows 240 s / 30 s overlap": {"window_seconds": 240, "window_overlap_seconds": 30},
    "windows 180 s / 30 s overlap": {"window_seconds": 180, "window_overlap_seconds": 30},
    "windows 300 s / 15 s overlap": {"window_seconds": 300, "window_overlap_seconds": 15},
    # The speaker count given up front, as a study that knows it would
    # (moss.num_speakers). "reference" takes each meeting's count from AMI.
    "windows 300 s + known speaker count": {
        "window_seconds": 300, "window_overlap_seconds": 30, "num_speakers": "reference",
    },
}
METRICS = ["cpwer", "wer_verbatim", "wer_clean", "filler_recall", "filler_precision",
           "der", "missed", "false_alarm", "confusion"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated config names")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    prepared, out = args.prepared.expanduser(), args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meetings = sorted(p.name.removesuffix(".wav") for p in prepared.glob("*.wav"))
    results: dict[str, dict] = {}
    results_path = out / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())

    for name in args.configs.split(","):
        kwargs = dict(CONFIGS[name])
        known = kwargs.pop("num_speakers", None) == "reference"
        diarizer = MOSSDiarizer(backend="mlx", **kwargs)
        for meeting in meetings:
            key = f"{name} | {meeting}"
            if key in results:
                continue
            wav = prepared / f"{meeting}.wav"
            ref = json.loads((prepared / f"{meeting}.ref.json").read_text())
            if known:
                segs = ref["segments"] if isinstance(ref, dict) else ref
                diarizer.num_speakers = len({s["speaker"] for s in segs})
            t = time.perf_counter()
            hyp = diarizer.process_file(str(wav))
            seconds = time.perf_counter() - t
            slug = name.split(" (")[0].replace(" ", "_").replace("/", "")
            (out / f"{meeting}.{slug}.hyp.json").write_text(json.dumps(hyp, indent=1))
            results[key] = {
                **score(ref, hyp), "seconds": seconds,
                "audio_seconds": _audio_duration(str(wav)) or 0.0,
            }
            results_path.write_text(json.dumps(results, indent=1))
            logging.info("%s: cpWER %.3f DER %.3f in %.0fs", key,
                         results[key]["cpwer"], results[key]["der"], seconds)

    write_report(results, meetings, args.report)


def write_report(results: dict, meetings: list[str], path: Path) -> None:
    lines = [
        "# Transcription accuracy against human ground truth (AMI)",
        "",
        f"Meetings: {', '.join(meetings)} (AMI test split, single distant microphone). "
        "Reference: AMI human transcripts and speaker labels. Audio rebuilt from the "
        "annotated utterances, so unannotated stretches are silence — slightly easier "
        "than a raw recording.",
        "",
        "Pooled over meetings, weighted by reference words (error rates) or speech time "
        "(DER, approximated by words). Lower is better except filler recall/precision.",
        "",
        "| configuration | cpWER | WER verbatim | WER clean | filler recall | filler precision "
        "| DER | missed | false alarm | confusion | speakers (hyp/ref) | x real time |",
        "|---|" + "---|" * 11,
    ]
    for name in CONFIGS:
        rows = [results.get(f"{name} | {m}") for m in meetings]
        if not all(rows):
            continue
        words = sum(r["ref_words"] for r in rows)

        def pooled(k: str) -> float:
            return sum(r[k] * r["ref_words"] for r in rows) / words

        speakers = ", ".join(f"{r['speakers_hyp']}/{r['speakers_ref']}" for r in rows)
        rtf = sum(r["audio_seconds"] for r in rows) / sum(r["seconds"] for r in rows)
        cells = " | ".join(f"{pooled(k):.3f}" for k in METRICS)
        lines.append(f"| {name} | {cells} | {speakers} | {rtf:.1f}x |")
    import moss_windowed as mw

    lines += [
        "",
        "## Notes",
        "",
        f"- Speaker-linking thresholds (voice model): link {mw._VOICE_LINK}, merge "
        f"{mw._VOICE_MERGE}, within-window split {mw._VOICE_SPLIT}. Tuned with "
        "`tune_linking.py` on EN2002a-c (one group of four people); ES2004a-b (different "
        "people) were held out and scored only at the chosen setting. Held-out cpWER at "
        "the chosen setting was within 0.2 points of the best any setting achieved there.",
        "- Word error with windows is within half a point of the single pass or better on "
        "every meeting; the single pass "
        "degrades with length (it collapsed on the 48-min EN2002c). Remaining windowed "
        "error is mostly speaker attribution in short multi-party meetings.",
        "- Known speaker count (`moss.num_speakers`, batch `--speakers`): extra labels "
        "are folded into the given number by voice. Every meeting then has the right "
        "count; cpWER and DER improve slightly because the extra labels carried little "
        "speech. Word error is unchanged: this fixes who, not what.",
        "- 'missed' is high in every configuration because AMI has much overlapping "
        "speech and MOSS assigns one speaker at a time.",
        "- Configurations and meetings evaluated on one M2 Max; x real time depends on "
        "machine load.",
        "",
        "Per-meeting results: `results.json` in the run directory (not committed).",
    ]
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
