"""Check each speaker label voice by voice, and split labels that hold two people.

MOSS labels speakers per 5-minute window. When someone first speaks partway
through a window (a late joiner in a group discussion), MOSS can give them the
label of a person already talking in that window, and window linking then
carries the merge through the whole file. Window-level voice prints cannot see
this, because they average the two voices together.

On AMI meetings with a joiner added, the bigger problem turned out to be
fragmentation: one person under two labels (segment-level voice similarity
0.70, while different people stayed below 0.6), and labels holding part of
another person's speech. So three steps run, in this order:

1. merge labels whose voices match at MERGE_SIMILARITY or above;
2. move a segment to another label when its voice matches that label better
   by REASSIGN_MARGIN (un-mixing labels), then merge again;
3. split a label whose segments form two different voice groups.

This looks at each segment. Segments of at least ``MIN_SEGMENT_S``
get their own voice embedding (WeSpeaker, see voice_embedder.py); stretches
where the overlap model hears two people at once are left out, since their
embedding mixes voices. A label whose segments form two clearly different
voice groups is split: the smaller group joins another speaker if its voice
matches one, or becomes a new speaker. Short segments follow the nearest
embedded segment of the same original label.

Thresholds were chosen on AMI EN2002a-c (with and without a joiner added
mid-window) and checked once on ES2004a-b (evals/ground_truth/late_joiner_eval.py,
evals/reports/group_speakers.md): speaker confusion there fell from 13.1% to
2.5% and from 12.1% to 1.0% with a late joiner, and from 6.8% to 1.2% on an
unaltered meeting, with no meeting worse.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("ClinicalWhisper")

SAMPLE_RATE = 16000
MIN_SEGMENT_S = 1.5
MAX_EMBED_S = 8.0
# Two voice groups inside one label are split when their centroids are less
# similar than this, and each group holds at least MIN_GROUP_S of speech.
SPLIT_SIMILARITY = 0.40
MIN_GROUP_S = 20.0
# A split-off group joins an existing speaker at or above this similarity.
JOIN_SIMILARITY = 0.55
# Labels whose voices match at least this strongly are one person.
MERGE_SIMILARITY = 0.60
# A segment moves to another label when its voice matches that label better
# than its own by this much (None: never).
REASSIGN_MARGIN: Optional[float] = 0.10
# Segments with more than this share of detected crosstalk are not embedded.
MAX_OVERLAP_SHARE = 0.3


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _overlap_share(seg: dict, regions: list[tuple[float, float]]) -> float:
    dur = seg["end"] - seg["start"]
    if dur <= 0 or not regions:
        return 0.0
    shared = sum(max(0.0, min(seg["end"], b) - max(seg["start"], a)) for a, b in regions)
    return shared / dur


def embed_segments(segments: list[dict], audio: np.ndarray, embedder,
                   overlap: Optional[list[tuple[float, float]]] = None) -> dict[int, np.ndarray]:
    """Unit voice embedding per usable segment index."""
    out: dict[int, np.ndarray] = {}
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        if dur < MIN_SEGMENT_S or _overlap_share(seg, overlap or []) > MAX_OVERLAP_SHARE:
            continue
        mid = (seg["start"] + seg["end"]) / 2
        half = min(dur, MAX_EMBED_S) / 2
        a, b = int((mid - half) * SAMPLE_RATE), int((mid + half) * SAMPLE_RATE)
        clip = audio[max(0, a):min(len(audio), b)]
        if len(clip) < MIN_SEGMENT_S * SAMPLE_RATE * 0.9:
            continue
        out[i] = _unit(np.asarray(embedder.embed(clip), dtype=np.float64))
    return out


def _centroid(idx: list[int], emb: dict[int, np.ndarray], dur: dict[int, float]) -> np.ndarray:
    return _unit(sum(emb[i] * dur[i] for i in idx))


def _two_groups(idx: list[int], emb: dict[int, np.ndarray]) -> tuple[list[int], list[int]]:
    from scipy.cluster.hierarchy import fcluster, linkage

    x = np.stack([emb[i] for i in idx])
    labels = fcluster(linkage(x, method="average", metric="cosine"), 2, criterion="maxclust")
    a = [i for i, lab in zip(idx, labels) if lab == 1]
    b = [i for i, lab in zip(idx, labels) if lab == 2]
    return a, b


def _centroids(segs: list[dict], emb: dict[int, np.ndarray], dur: dict[int, float]) -> dict[str, np.ndarray]:
    by_label: dict[str, list[int]] = {}
    for i in emb:
        by_label.setdefault(segs[i]["speaker"], []).append(i)
    return {lab: _centroid(idx, emb, dur) for lab, idx in by_label.items()}


def merge_labels(segs: list[dict], emb: dict[int, np.ndarray], dur: dict[int, float],
                 threshold: float, min_s: float = 5.0) -> list[dict]:
    """Merge labels (with at least ``min_s`` of embedded speech) whose voices match."""
    changes = []
    while True:
        talk: dict[str, float] = {}
        for i in emb:
            talk[segs[i]["speaker"]] = talk.get(segs[i]["speaker"], 0.0) + dur[i]
        cents = {k: v for k, v in _centroids(segs, emb, dur).items() if talk.get(k, 0) >= min_s}
        labs = sorted(cents)
        pairs = [(float(cents[a] @ cents[b]), a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
        if not pairs:
            return changes
        sim, a, b = max(pairs)
        if sim < threshold:
            return changes
        keep, drop = (a, b) if talk[a] >= talk[b] else (b, a)
        moved = 0.0
        for j, sg in enumerate(segs):
            if sg["speaker"] == drop:
                sg["speaker"] = keep
                moved += dur[j]
        changes.append({"from": drop, "to": keep, "joined_existing": True, "seconds": round(moved, 1),
                        "first_s": round(min((sg["start"] for sg in segs if sg["speaker"] == keep), default=0), 1),
                        "similarity": round(sim, 3), "kind": "merged"})


def reassign(segs: list[dict], emb: dict[int, np.ndarray], dur: dict[int, float],
             margin: float, min_label_s: float = 20.0) -> list[dict]:
    """Move segments whose voice clearly matches another label better than their own."""
    talk: dict[str, float] = {}
    for i in emb:
        talk[segs[i]["speaker"]] = talk.get(segs[i]["speaker"], 0.0) + dur[i]
    cents = {k: v for k, v in _centroids(segs, emb, dur).items() if talk.get(k, 0) >= min_label_s}
    moves: dict[tuple[str, str], float] = {}
    for i, v in emb.items():
        own = segs[i]["speaker"]
        if own not in cents or len(cents) < 2:
            continue
        # The segment's own label without the segment itself.
        rest = cents[own] * talk[own] - v * dur[i]
        n = np.linalg.norm(rest)
        own_sim = float(v @ (rest / n)) if n > 0 else 1.0
        best = max((k for k in cents if k != own), key=lambda k: float(v @ cents[k]))
        if float(v @ cents[best]) - own_sim >= margin:
            segs[i]["speaker"] = best
            moves[(own, best)] = moves.get((own, best), 0.0) + dur[i]
    return [{"from": a, "to": b, "joined_existing": True, "seconds": round(t, 1), "first_s": None,
             "similarity": None, "kind": "moved"} for (a, b), t in moves.items()]


def refine(segments: list[dict], emb: dict[int, np.ndarray], merge: float = MERGE_SIMILARITY,
           margin: Optional[float] = REASSIGN_MARGIN, split_similarity: float = SPLIT_SIMILARITY,
           join_similarity: float = JOIN_SIMILARITY, min_group_s: float = MIN_GROUP_S
           ) -> tuple[list[dict], list[dict]]:
    """Merge, un-mix and split speaker labels by voice (see the module notes)."""
    segs = [dict(s) for s in segments]
    dur = {i: s["end"] - s["start"] for i, s in enumerate(segs)}
    changes = merge_labels(segs, emb, dur, merge)
    if margin is not None:
        changes += reassign(segs, emb, dur, margin)
        changes += merge_labels(segs, emb, dur, merge)
    segs, split = recheck(segs, emb, split_similarity, join_similarity, min_group_s)
    changes += split
    if changes:
        log.info("Speaker check: %d change(s) by voice.", len(changes))
    return segs, changes


def recheck(segments: list[dict], emb: dict[int, np.ndarray],
            split_similarity: float = SPLIT_SIMILARITY, join_similarity: float = JOIN_SIMILARITY,
            min_group_s: float = MIN_GROUP_S, rounds: int = 3) -> tuple[list[dict], list[dict]]:
    """Segments with mixed labels split; and one record per change made.

    Records hold the old and new label, the seconds moved and when the moved
    voice first speaks, e.g. ``{"from": "S02", "to": "S05", "seconds": 312.4,
    "first_s": 421.0, "joined_existing": False}``.
    """
    segs = [dict(s) for s in segments]
    dur = {i: s["end"] - s["start"] for i, s in enumerate(segs)}
    changes: list[dict] = []
    for _ in range(rounds):
        by_label: dict[str, list[int]] = {}
        for i in emb:
            by_label.setdefault(segs[i]["speaker"], []).append(i)
        centroids = {lab: _centroid(idx, emb, dur) for lab, idx in by_label.items()}
        changed = False
        for lab, idx in sorted(by_label.items()):
            if len(idx) < 4 or sum(dur[i] for i in idx) < 2 * min_group_s:
                continue
            a, b = _two_groups(idx, emb)
            if not a or not b:
                continue
            sa, sb = sum(dur[i] for i in a), sum(dur[i] for i in b)
            if min(sa, sb) < min_group_s:
                continue
            ca, cb = _centroid(a, emb, dur), _centroid(b, emb, dur)
            if float(ca @ cb) >= split_similarity:
                continue
            keep, move = (a, b) if sa >= sb else (b, a)
            moved = _centroid(move, emb, dur)
            others = {o: float(moved @ c) for o, c in centroids.items() if o != lab}
            best = max(others, key=others.get, default=None)
            if best is not None and others[best] >= join_similarity:
                target, existing = best, True
            else:
                used = {s["speaker"] for s in segs}
                n = len(used) + 1
                while f"S{n:02d}" in used:
                    n += 1
                target, existing = f"S{n:02d}", False
            # Short and unembedded segments of this label follow the nearest
            # embedded segment in time.
            anchors = sorted((segs[i]["start"], i in move) for i in idx)
            starts = np.array([t for t, _ in anchors])
            follows = [m for _, m in anchors]
            moved_idx = set(move)
            for j, s in enumerate(segs):
                if s["speaker"] != lab or j in emb:
                    continue
                k = int(np.argmin(np.abs(starts - s["start"])))
                if follows[k]:
                    moved_idx.add(j)
            for j in moved_idx:
                segs[j]["speaker"] = target
            changes.append({"kind": "split", "from": lab, "to": target, "joined_existing": existing,
                            "seconds": round(sum(dur[j] for j in moved_idx), 1),
                            "first_s": round(min(segs[j]["start"] for j in moved_idx), 1),
                            "similarity": round(float(ca @ cb), 3)})
            changed = True
            break  # centroids changed; recompute before the next split
        if not changed:
            break
    if changes:
        log.info("Speaker check: %s", "; ".join(
            f"{c['seconds']:.0f}s from {c['from']} to {c['to']}" for c in changes))
    return segs, changes
