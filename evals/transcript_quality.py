"""Compare MOSS transcripts against an independent Whisper transcript.

Speed changes to transcription must not cost accuracy. With no hand-made
ground truth, this uses two references:

* **Whisper large-v3** — a different model family, so agreement with it is
  independent evidence. Reported as recall of Whisper's words (fillers
  removed, since Whisper tends to drop them), per time block.
* **Single-pass MOSS** — the configuration being replaced. Fillers and
  speaker labels are compared against it, because Whisper has neither.

    uv run python evals/transcript_quality.py whisper.json single.json \\
        windowed_300=w300.json windowed_240=w240.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

FILLERS = re.compile(r"\b(um+|uh+|mm+|hmm+|erm|mhm|uh-huh)\b", re.I)


def words(segments: list[dict], start: float = 0, end: float = float("inf")) -> list[str]:
    text = " ".join(s["text"] for s in segments if start <= s["start"] < end)
    text = FILLERS.sub(" ", text.lower())
    return re.sub(r"[^\w\s']", " ", text).split()


def recall(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return float("nan")
    sm = difflib.SequenceMatcher(None, reference, candidate, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(reference)


def speaker_agreement(ref: list[dict], cand: list[dict], step: float = 0.25) -> float:
    """Share of speech time with matching labels, after the best one-to-one
    relabelling of candidate speakers onto reference speakers."""
    end = max(s["end"] for s in ref)
    ts = np.arange(0, end, step)

    def labels(segs):
        out = np.full(len(ts), "", dtype=object)
        for s in segs:
            out[(ts >= s["start"]) & (ts < s["end"])] = s["speaker"]
        return out

    a, b = labels(ref), labels(cand)
    both = (a != "") & (b != "")
    ra, rb = sorted(set(a[both])), sorted(set(b[both]))
    counts = np.zeros((len(rb), len(ra)))
    for i, x in enumerate(rb):
        for j, y in enumerate(ra):
            counts[i, j] = np.sum((b[both] == x) & (a[both] == y))
    rows, cols = linear_sum_assignment(-counts)
    return counts[rows, cols].sum() / both.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("whisper")
    ap.add_argument("single")
    ap.add_argument("candidates", nargs="+", help="name=path")
    ap.add_argument("--block", type=float, default=180.0)
    args = ap.parse_args()

    whisper = json.load(open(args.whisper))
    runs = {"single": json.load(open(args.single))}
    runs.update({c.split("=", 1)[0]: json.load(open(c.split("=", 1)[1])) for c in args.candidates})

    end = max(s["end"] for s in whisper if s.get("end") is not None)
    print(f"{'block':>11} " + " ".join(f"{k:>14}" for k in runs))
    for a in np.arange(0, end, args.block):
        ref = words(whisper, a, a + args.block)
        if len(ref) < 20:
            continue
        cells = [f"{recall(ref, words(r, a, a + args.block)):14.3f}" for r in runs.values()]
        print(f"{a / 60:4.0f}-{(a + args.block) / 60:<4.0f}min " + " ".join(cells))

    ref_all = words(whisper)
    single_fill = len(FILLERS.findall(" ".join(s["text"] for s in runs["single"])))
    print()
    for name, segs in runs.items():
        fill = len(FILLERS.findall(" ".join(s["text"] for s in segs)))
        line = (f"{name:>14}: whisper-word recall {recall(ref_all, words(segs)):.3f}, "
                f"fillers {fill} ({fill / max(single_fill, 1):.0%} of single), "
                f"speakers {len({s['speaker'] for s in segs})}, "
                f"covers to {max(s['end'] for s in segs) / 60:.1f} min")
        if name != "single":
            line += f", speaker agreement vs single {speaker_agreement(runs['single'], segs):.3f}"
        print(line)


if __name__ == "__main__":
    main()
