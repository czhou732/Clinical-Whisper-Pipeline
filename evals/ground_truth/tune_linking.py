"""Tune cross-window speaker thresholds against AMI human labels.

Transcription is the slow part and does not depend on the thresholds, so this
replays saved per-window output (written when ``CW_DUMP_WINDOWS`` is set) through
the linking, consolidation and merge steps at every threshold pair, and scores
each result against the human reference.

    CW_DUMP_WINDOWS=~/Developer/datasets/ami/windows \\
        uv run python evals/ground_truth/run_ami.py ... --configs "windows 300 s / 30 s overlap"
    uv run python evals/ground_truth/tune_linking.py \\
        ~/Developer/datasets/ami/windows ~/Developer/datasets/ami/prepared
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from moss_chunking import Window, consolidate_speakers, link_speakers, merge_windows  # noqa: E402
from score import cpwer, der  # noqa: E402

LINK_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MERGE_GRID = [0.50, 0.60, 0.65, 0.70, 0.80, 1.01]  # 1.01 = never merge
SPLIT_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, None]  # None = never merge within a window


def load_dump(path: Path):
    d = json.loads(path.read_text())
    windows = [Window(*w) for w in d["windows"]]
    embeddings = [
        {spk: (np.array(vec), talk) for spk, (vec, talk) in emb.items()} for emb in d["embeddings"]
    ]
    return windows, d["segments"], embeddings


def evaluate(dumps: dict, refs: dict, link_at: float, merge_at: float, center: bool,
             split_at=None) -> dict:
    errors = words = der_num = der_den = 0.0
    speaker_err = 0
    for meeting, (windows, segments, embeddings) in dumps.items():
        maps = link_speakers(windows, segments, embeddings, link_at, center=center,
                             split_threshold=split_at)
        maps = consolidate_speakers(embeddings, maps, merge_at, center=center,
                                    split_threshold=split_at)
        hyp = merge_windows(windows, segments, maps)
        ref = refs[meeting]
        n = sum(len(s["text"].split()) for s in ref)
        errors += cpwer(ref, hyp) * n
        words += n
        dur = sum(s["end"] - s["start"] for s in ref)
        der_num += der(ref, hyp)["der"] * dur
        der_den += dur
        speaker_err += abs(len({s["speaker"] for s in hyp}) - len({s["speaker"] for s in ref}))
    return {"cpwer": errors / words, "der": der_num / der_den, "speaker_count_error": speaker_err}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", type=Path)
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--center", action="store_true", help="for encoder-feature embeddings")
    ap.add_argument("--tune-on", default="", help="meeting-name prefix to tune on (default: all)")
    ap.add_argument("--hold-out", default="", help="meeting-name prefix scored only at the chosen setting")
    args = ap.parse_args()

    all_dumps = {p.name.split(".")[0]: load_dump(p)
                 for p in sorted(args.dumps.expanduser().glob("*.windows.json"))}
    refs = {m: json.loads((args.prepared.expanduser() / f"{m}.ref.json").read_text()) for m in all_dumps}
    held = {m: d for m, d in all_dumps.items() if args.hold_out and m.startswith(args.hold_out)}
    dumps = {m: d for m, d in all_dumps.items()
             if m not in held and (not args.tune_on or m.startswith(args.tune_on))}
    print(f"tuning on: {', '.join(dumps)}" + (f" | held out: {', '.join(held)}" if held else ""))

    results = []
    for link_at, merge_at, split_at in itertools.product(LINK_GRID, MERGE_GRID, SPLIT_GRID):
        r = evaluate(dumps, refs, link_at, merge_at, args.center, split_at)
        results.append((r["cpwer"], r["der"], r["speaker_count_error"], link_at, merge_at,
                        split_at if split_at is not None else 9.0))
    results.sort()
    print(f"{'link':>5} {'merge':>5} {'split':>5} {'cpWER':>7} {'DER':>7} {'|spk err|':>9}")
    for c, d, e, link_at, merge_at, split_at in results[:15]:
        print(f"{link_at:5.2f} {merge_at:5.2f} {split_at:5.2f} {c:7.3f} {d:7.3f} {e:9d}")
    c, d, e, link_at, merge_at, split_at = results[0]
    print(f"\nbest by cpWER: link {link_at:.2f}, merge {merge_at:.2f}, split {split_at:.2f} "
          f"-> cpWER {c:.3f}, DER {d:.3f}, speaker-count error {e}  (split 9.00 = never)")
    if held:
        chosen = None if split_at >= 9 else split_at
        for label, args_ in (("chosen", (link_at, merge_at, chosen)), ("no within-window merge", (link_at, merge_at, None))):
            r = evaluate(held, refs, *args_[:2], args.center, args_[2])
            print(f"held-out ({label}): cpWER {r['cpwer']:.3f}, DER {r['der']:.3f}, "
                  f"speaker-count error {r['speaker_count_error']}")


if __name__ == "__main__":
    main()
