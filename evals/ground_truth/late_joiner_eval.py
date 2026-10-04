"""Tune and test speaker_check.py (late joiners) and overlap marking on AMI.

Inputs are MOSS transcripts already made by run_ami.py, for the original
meetings and for the late-joiner versions (late_joiner_prepare.py). Voice
embeddings and overlap regions are cached next to the transcripts.

* Tuning: EN2002a-c, late-joiner and original versions. A setting must lower
  speaker error on the late-joiner meetings without raising it on the
  originals, where nobody joins late.
* Test: ES2004a-b, scored once at the chosen setting.
* Overlap: precision and recall of detected crosstalk against AMI's
  annotated overlaps, threshold chosen on EN2002, reported on ES2004.

    python evals/ground_truth/late_joiner_eval.py --report evals/reports/group_speakers.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import speaker_check  # noqa: E402
from score import score  # noqa: E402

AMI = Path.home() / "Developer" / "datasets" / "ami"
SLUG = "windows_300_s__30_s_overlap"
TUNE = ["EN2002a", "EN2002b", "EN2002c"]
TEST = ["ES2004a", "ES2004b"]
MERGES = [0.60, 0.65, 0.70, 0.75, 9.0]  # 9.0: never merge
MARGINS = [0.10, 0.15, 0.20, 0.30, None]
SPLITS = [0.40, 0.50]


def files(meeting: str, late: bool) -> tuple[Path, Path, Path]:
    if late:
        base = AMI / "late_joiner"
        return (base / f"{meeting}_late.wav", base / f"{meeting}_late.ref.json",
                AMI / "late_joiner_runs" / f"{meeting}_late.{SLUG}.hyp.json")
    return (AMI / "prepared" / f"{meeting}.wav", AMI / "prepared" / f"{meeting}.ref.json",
            AMI / "runs" / f"{meeting}.{SLUG}.hyp.json")


_TOOLS: dict = {}


def tools():
    if not _TOOLS:
        from overlap_detector import OverlapDetector
        from voice_embedder import VoiceEmbedder
        _TOOLS["emb"], _TOOLS["ov"] = VoiceEmbedder.load(), OverlapDetector.load()
    return _TOOLS["emb"], _TOOLS["ov"]


def cached(meeting: str, late: bool) -> dict:
    wav, ref_p, hyp_p = files(meeting, late)
    cache = hyp_p.with_suffix(".check.npz")
    hyp = json.loads(hyp_p.read_text())
    ref = json.loads(ref_p.read_text())
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        ov = [tuple(r) for r in z["overlap"]]
        emb = {int(i): v for i, v in zip(z["idx"], z["emb"])}
        probs = z["probs"]
        frame_s = float(z["frame_s"])
    else:
        embedder, detector = tools()
        audio, _ = sf.read(str(wav), dtype="float32")
        probs, _, frame_s = detector.probabilities(audio)
        ov = _regions(probs, frame_s, 0.5)
        emb = speaker_check.embed_segments(hyp, audio, embedder, ov)
        idx = np.array(sorted(emb), dtype=np.int64)
        np.savez(cache, overlap=np.array(ov, dtype=np.float64).reshape(-1, 2), idx=idx,
                 emb=np.stack([emb[i] for i in idx]) if len(idx) else np.zeros((0, 256)),
                 probs=probs, frame_s=frame_s)
    return {"hyp": hyp, "ref": ref, "emb": emb, "overlap": ov, "probs": probs, "frame_s": frame_s}


def _regions(probs: np.ndarray, frame_s: float, th: float) -> list[tuple[float, float]]:
    on = np.concatenate([[False], probs >= th, [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    return [(a * frame_s, b * frame_s) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * frame_s >= 0.3]


def overlap_pr(d: dict, th: float) -> tuple[float, float, float]:
    probs, fs = d["probs"], d["frame_s"]
    act = np.zeros(len(probs))
    for r in d["ref"]:
        a, b = int(r["start"] / fs), int(r["end"] / fs)
        act[a:min(b, len(act))] += 1
    ref_ov, hyp = act >= 2, probs >= th
    tp = float((ref_ov & hyp).sum())
    return tp, float(hyp.sum()), float(ref_ov.sum())


def run(d: dict, merge: float, margin, split: float) -> dict:
    segs, changes = speaker_check.refine(d["hyp"], d["emb"], merge, margin, split)
    s = score(d["ref"], segs)
    talk: dict[str, float] = {}
    for x in segs:
        talk[x["speaker"]] = talk.get(x["speaker"], 0.0) + x["end"] - x["start"]
    return {"der": s["der"], "confusion": s["confusion"], "cpwer": s["cpwer"],
            "speakers": len(talk), "substantial": sum(t >= 30 for t in talk.values()),
            "changes": len(changes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    data = {(m, late): cached(m, late) for m in TUNE + TEST for late in (True, False)}
    base = {k: score(v["ref"], v["hyp"]) for k, v in data.items()}

    # Tuning: the late-joiner meetings must improve, the originals must not get worse.
    best = None
    orig_base = np.mean([base[(m, False)]["confusion"] for m in TUNE])
    for merge, margin, split in itertools.product(MERGES, MARGINS, SPLITS):
        late = np.mean([run(data[(m, True)], merge, margin, split)["confusion"] for m in TUNE])
        orig = np.mean([run(data[(m, False)], merge, margin, split)["confusion"] for m in TUNE])
        if orig > orig_base + 0.002:
            continue
        # Both kinds of meeting count; a setting must not trade one for the other.
        if best is None or late + orig < best[0]:
            best = (late + orig, merge, margin, split)
    _, merge, margin, split = best

    ths = [0.3, 0.4, 0.5, 0.6]
    def f1(meetings, th):
        tp = hp = rp = 0.0
        for m in meetings:
            a, b, c = overlap_pr(data[(m, False)], th)
            tp, hp, rp = tp + a, hp + b, rp + c
        p, r = tp / max(hp, 1), tp / max(rp, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)
    ov_th = max(ths, key=lambda t: f1(TUNE, t)[2])

    lines = ["# Speaker labels in group discussions: late joiners and crosstalk", "",
             "AMI test meetings (single distant microphone, audio rebuilt from annotated utterances).",
             "Late-joiner versions: one participant's speech removed before 7 min, so they join",
             "mid-window (late_joiner_prepare.py). MOSS transcripts made once with the current",
             "pipeline (300 s windows); speaker_check.py and overlap_detector.py run on top.", "",
             f"Chosen on EN2002a-c: merge labels at voice similarity {merge}, move a segment when",
             f"another label matches it better by {margin}, split below {split}. Overlap threshold {ov_th}.", "",
             "## Speaker error (DER confusion, 0.25 s collar) and word error by speaker (cpWER)", "",
             "| meeting | set | confusion before | after | cpWER before | after | labels before / after (with 30 s+ talk) / true people |",
             "|---|---|---|---|---|---|---|"]
    for m in TUNE + TEST:
        for late in (True, False):
            d = data[(m, late)]
            r = run(d, merge, margin, split)
            b = base[(m, late)]
            true_n = len({x["speaker"] for x in d["ref"]})
            lines.append(f"| {m}{' late joiner' if late else ''} | {'tuning' if m in TUNE else 'test'} | "
                         f"{b['confusion']:.3f} | {r['confusion']:.3f} | {b['cpwer']:.3f} | {r['cpwer']:.3f} | "
                         f"{len({x['speaker'] for x in d['hyp']})} / {r['speakers']} ({r['substantial']}) / {true_n} |")
    lines += ["", "## Crosstalk detection (frames where two or more people talk, AMI annotation)", "",
              "| meetings | precision | recall | F1 |", "|---|---|---|---|"]
    for name, ms in (("EN2002a-c (tuning)", TUNE), ("ES2004a-b (test)", TEST)):
        p, r, f = f1(ms, ov_th)
        lines.append(f"| {name} | {p:.2f} | {r:.2f} | {f:.2f} |")
    lines += ["", "Labels with under 30 s of talk are mostly a backchannel (\"yeah\") given its own",
              "label; the app flags labels under 10 s as possibly one person split in two.",
              "", "Read with care: rebuilt audio has silence where nobody was annotated, which is",
              "easier than a raw recording; five meetings, one joiner each."]
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
