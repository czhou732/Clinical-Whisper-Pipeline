"""Score a diarized transcript against a human reference.

Both files are ``[{"speaker", "start", "end", "text"}]``. Metrics:

* **cpWER** — concatenated minimum-permutation WER (CHiME-6): each speaker's
  words are concatenated, hypothesis speakers are matched to reference speakers
  by optimal assignment, and errors are counted over the matched streams. It
  measures *who said what* in one number.
* **WER** — speaker-agnostic, verbatim (fillers kept) and clean (fillers removed).
* **Filler precision / recall** — from the verbatim alignment; hesitancy
  measures depend on these tokens surviving.
* **DER** — diarization error rate on 10 ms frames with a 0.25 s collar around
  reference boundaries, split into missed speech, false alarm and speaker
  confusion, after optimal speaker mapping.
* **Speaker count** — hypothesis vs reference.

    uv run python evals/ground_truth/score.py EN2002a.ref.json EN2002a.hyp.json
"""

from __future__ import annotations

import argparse
import json
import re

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

# Spelling variants mapped to one token so style differences are not errors.
_FILLER_CANON = {
    "um": "um", "umm": "um", "erm": "um", "em": "um",
    "uh": "uh", "uhh": "uh", "er": "uh",
    "hmm": "hmm", "hm": "hmm", "mm": "hmm", "mmm": "hmm",
    "mhm": "mmhmm", "mmhmm": "mmhmm", "mmhm": "mmhmm", "uhhuh": "mmhmm",
}
FILLERS = set(_FILLER_CANON.values())

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen " \
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _number_words(n: int) -> str:
    """Spell out 0-9999 the way AMI's transcribers wrote numbers."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _number_words(rest))
    rest = n % 1000
    return _number_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + _number_words(rest))


def normalize(text: str) -> list[str]:
    """Lowercase, spell out small numbers, drop punctuation, unify fillers."""
    text = text.lower()
    text = re.sub(r"\b\d{1,4}\b", lambda m: _number_words(int(m.group())), text)
    text = re.sub(r"(\w)-(\w)", r"\1\2", text)  # mm-hmm -> mmhmm, uh-huh -> uhhuh
    text = re.sub(r"[^\w\s']", " ", text)
    return [_FILLER_CANON.get(w, w) for w in text.split()]


def _words(segments: list[dict], drop_fillers: bool = False) -> list[str]:
    out: list[str] = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        out.extend(w for w in normalize(seg["text"]) if not (drop_fillers and w in FILLERS))
    return out


def _errors(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return o.substitutions + o.deletions + o.insertions


def wer(ref: list[dict], hyp: list[dict], drop_fillers: bool = False) -> float:
    r, h = _words(ref, drop_fillers), _words(hyp, drop_fillers)
    return _errors(r, h) / max(len(r), 1)


def cpwer(ref: list[dict], hyp: list[dict]) -> float:
    def streams(segs):
        by: dict[str, list[dict]] = {}
        for s in segs:
            by.setdefault(s["speaker"], []).append(s)
        return {k: _words(v) for k, v in by.items()}

    rs, hs = streams(ref), streams(hyp)
    rk, hk = list(rs), list(hs)
    n = max(len(rk), len(hk))
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = rs[rk[i]] if i < len(rk) else []
            h = hs[hk[j]] if j < len(hk) else []
            cost[i, j] = _errors(r, h)
    rows, cols = linear_sum_assignment(cost)
    total_ref = sum(len(v) for v in rs.values())
    return cost[rows, cols].sum() / max(total_ref, 1)


def filler_precision_recall(ref: list[dict], hyp: list[dict]) -> tuple[float, float]:
    r, h = _words(ref), _words(hyp)
    o = jiwer.process_words(" ".join(r), " ".join(h))
    matched = 0
    for chunk in o.alignments[0]:
        if chunk.type == "equal":
            matched += sum(1 for w in r[chunk.ref_start_idx:chunk.ref_end_idx] if w in FILLERS)
    n_ref = sum(w in FILLERS for w in r)
    n_hyp = sum(w in FILLERS for w in h)
    return matched / max(n_hyp, 1), matched / max(n_ref, 1)


def der(ref: list[dict], hyp: list[dict], collar: float = 0.25, step: float = 0.01) -> dict:
    end = max(max(s["end"] for s in ref), max((s["end"] for s in hyp), default=0))
    n = int(np.ceil(end / step)) + 1

    def activity(segs):
        labels = sorted({s["speaker"] for s in segs})
        act = np.zeros((n, len(labels)), dtype=bool)
        for s in segs:
            act[int(s["start"] / step):int(s["end"] / step), labels.index(s["speaker"])] = True
        return act

    R, H = activity(ref), activity(hyp)
    scored = np.ones(n, dtype=bool)
    c = int(collar / step)
    for s in ref:
        for edge in (s["start"], s["end"]):
            k = int(edge / step)
            scored[max(0, k - c):k + c] = False
    R, H = R[scored], H[scored]
    nr, nh = R.sum(1), H.sum(1)
    overlap = R.T.astype(np.int64) @ H.astype(np.int64)  # frames each pair co-talks
    rows, cols = linear_sum_assignment(-overlap)
    correct = overlap[rows, cols].sum()
    total = nr.sum()
    miss = np.maximum(nr - nh, 0).sum()
    fa = np.maximum(nh - nr, 0).sum()
    conf = np.minimum(nr, nh).sum() - correct
    return {
        "der": float((miss + fa + conf) / max(total, 1)),
        "missed": float(miss / max(total, 1)),
        "false_alarm": float(fa / max(total, 1)),
        "confusion": float(conf / max(total, 1)),
    }


def score(ref: list[dict], hyp: list[dict], collar: float = 0.25) -> dict:
    """All measures. ``collar=0`` matches the published AMI tables (no collar).

    For a table directly comparable to papers, cpWER and tcpWER should come
    from MeetEval itself; this cpWER is the same definition, kept dependency-free.
    """
    precision, recall = filler_precision_recall(ref, hyp)
    return {
        "cpwer": cpwer(ref, hyp),
        "wer_verbatim": wer(ref, hyp),
        "wer_clean": wer(ref, hyp, drop_fillers=True),
        "filler_precision": precision,
        "filler_recall": recall,
        **der(ref, hyp, collar=collar),
        "speakers_ref": len({s["speaker"] for s in ref}),
        "speakers_hyp": len({s["speaker"] for s in hyp}),
        "ref_words": len(_words(ref)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("hypothesis")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="DER collar in seconds (0 for the published AMI tables)")
    args = ap.parse_args()
    result = score(json.load(open(args.reference)), json.load(open(args.hypothesis)),
                   collar=args.collar)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, indent=1))
