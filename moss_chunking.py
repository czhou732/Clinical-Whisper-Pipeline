"""Windowed transcription support for MOSS: planning, speaker linking, merging.

Transcribing an hour in one pass is slow for a structural reason: every output
token attends over the whole audio prompt (~50k tokens), so decode speed falls
from ~50 tok/s at 4k context to ~9 tok/s at 50k. Splitting the audio into
equal-length overlapping windows keeps context short, and equal lengths mean
the windows can be decoded together as one batch with no padding.

The cost is that MOSS numbers speakers per window — ``S01`` in one window may
be ``S02`` in the next. :func:`link_speakers` maps window-local labels to
file-global ones using two kinds of evidence:

1. **Overlap agreement.** Adjacent windows transcribe the same stretch of
   audio; whoever is talking there must be the same person in both.
2. **Voice similarity.** Each speaker's mean audio-encoder embedding, compared
   against running per-speaker centroids. This covers speakers who are silent
   in an overlap, and speakers who return after a long absence.

Everything here is pure numpy so it can be tested without loading a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Seconds per audio-encoder output frame: Whisper's 20 ms frames, merged 4x.
FRAME_SECONDS = 0.08

# A speaker needs this much talk inside an overlap before the overlap alone
# decides their identity.
_MIN_OVERLAP_EVIDENCE_S = 2.0
_OVERLAP_AGREEMENT = 0.6
_SAMPLE_STEP_S = 0.1


@dataclass(frozen=True)
class Window:
    """A slice of the recording, in samples and seconds."""

    index: int
    start_sample: int
    n_samples: int
    sample_rate: int

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return (self.start_sample + self.n_samples) / self.sample_rate


def plan_windows(
    n_samples: int, sample_rate: int, window_s: float, overlap_s: float
) -> list[Window]:
    """Cover ``[0, n_samples)`` with equal-length windows overlapping by at
    least ``overlap_s``.

    Equal length matters: the windows are decoded as one batch, and identical
    sample counts give identical prompt lengths, so no padding or masking is
    needed. A recording shorter than one window is returned whole.
    """
    length = int(window_s * sample_rate)
    overlap = int(overlap_s * sample_rate)
    if n_samples <= length:
        return [Window(0, 0, n_samples, sample_rate)]
    if overlap >= length:
        raise ValueError("overlap must be shorter than the window")

    count = math.ceil((n_samples - overlap) / (length - overlap))
    # Stretch each window so the set spans the file exactly.
    length = math.ceil((n_samples + (count - 1) * overlap) / count)
    step = (n_samples - length) / (count - 1)
    return [
        Window(i, int(round(i * step)), length, sample_rate) for i in range(count)
    ]


def speaker_embeddings(
    segments: list[dict], features: np.ndarray
) -> dict[str, tuple[np.ndarray, float]]:
    """Mean encoder embedding and total talk time per local speaker.

    ``segments`` carry window-local times; ``features`` is ``(frames, dim)``
    at :data:`FRAME_SECONDS` per frame.
    """
    sums: dict[str, np.ndarray] = {}
    talk: dict[str, float] = {}
    for seg in segments:
        a = max(0, int(seg["start"] / FRAME_SECONDS))
        b = min(len(features), int(seg["end"] / FRAME_SECONDS))
        if b <= a:
            continue
        spk = seg["speaker"]
        sums[spk] = sums.get(spk, 0) + features[a:b].sum(axis=0)
        talk[spk] = talk.get(spk, 0.0) + (b - a) * FRAME_SECONDS
    return {
        spk: (sums[spk] / (talk[spk] / FRAME_SECONDS), talk[spk]) for spk in sums
    }


def _label_at(segments: list[dict], t: float, offset: float) -> Optional[str]:
    for seg in segments:
        if seg["start"] + offset <= t < seg["end"] + offset:
            return seg["speaker"]
    return None


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def link_speakers(
    windows: list[Window],
    window_segments: list[list[dict]],
    window_embeddings: list[dict[str, tuple[np.ndarray, float]]],
    similarity_threshold: float,
    center: bool = True,
    split_threshold: Optional[float] = None,
) -> list[dict[str, str]]:
    """Map each window's local speaker labels to file-global labels.

    ``split_threshold``: MOSS sometimes gives one person two labels inside a
    window (on AMI, 5-7 labels in windows with 4 people). Two labels in one
    window are normally kept apart, but a label whose voice matches an existing
    speaker at least this strongly joins it anyway. ``None`` never merges.

    ``center`` subtracts the mean embedding before comparing — needed for raw
    encoder features, which share a large common component; not for trained
    speaker embeddings.

    Returns one ``{local_label: global_label}`` dict per window. Global labels
    are ``S01``, ``S02``, ... in order of first appearance.
    """
    # Remove the component shared by every speaker (channel, room, content)
    # so that cosine similarity reflects voice differences.
    all_vecs = [v for emb in window_embeddings for v, _ in emb.values()]
    common = np.mean(all_vecs, axis=0) if (all_vecs and center) else 0.0

    centroids: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    mappings: list[dict[str, str]] = []

    def new_label() -> str:
        return f"S{len(centroids) + 1:02d}"

    for i, (win, segs, emb) in enumerate(zip(windows, window_segments, window_embeddings)):
        vecs = {spk: _unit(v - common) for spk, (v, _) in emb.items()}
        candidates: list[tuple[float, str, str]] = []

        if i > 0:
            prev_win, prev_segs, prev_map = windows[i - 1], window_segments[i - 1], mappings[i - 1]
            counts: dict[str, dict[str, float]] = {}
            t = win.start
            while t < prev_win.end:
                cur = _label_at(segs, t, win.start)
                prev = _label_at(prev_segs, t, prev_win.start)
                if cur is not None and prev is not None and prev in prev_map:
                    row = counts.setdefault(cur, {})
                    g = prev_map[prev]
                    row[g] = row.get(g, 0.0) + _SAMPLE_STEP_S
                t += _SAMPLE_STEP_S
            for spk, row in counts.items():
                total = sum(row.values())
                if total >= _MIN_OVERLAP_EVIDENCE_S:
                    g, n = max(row.items(), key=lambda kv: kv[1])
                    if n / total >= _OVERLAP_AGREEMENT:
                        # Overlap evidence outranks any voice similarity.
                        candidates.append((2.0 + n / total, spk, g))

        for spk, v in vecs.items():
            for g, c in centroids.items():
                sim = float(np.dot(v, _unit(c)))
                if sim >= similarity_threshold:
                    candidates.append((sim, spk, g))

        mapping: dict[str, str] = {}
        taken: set[str] = set()
        for _, spk, g in sorted(candidates, reverse=True):
            if spk not in mapping and g not in taken:
                mapping[spk] = g
                taken.add(g)

        # Unmatched speakers, in order of appearance within the window. A very
        # close voice match joins that speaker even if it is already taken in
        # this window (MOSS split one person); otherwise it is a new speaker.
        for seg in sorted(segs, key=lambda s: s["start"]):
            spk = seg["speaker"]
            if spk in mapping:
                continue
            if split_threshold is not None and spk in vecs:
                pool = {g: _unit(c) for g, c in centroids.items() if np.any(c)}
                pool.update({mapping[o]: vecs[o] for o in mapping if o in vecs})
                best = max(pool, key=lambda g: float(np.dot(vecs[spk], pool[g])), default=None)
                if best is not None and float(np.dot(vecs[spk], pool[best])) >= split_threshold:
                    mapping[spk] = best
                    continue
            g = new_label()
            mapping[spk] = g
            centroids[g] = np.zeros_like(next(iter(vecs.values()))) if vecs else np.zeros(1)
            weights[g] = 0.0

        for spk, (_, talk) in emb.items():
            g = mapping[spk]
            centroids[g] = centroids[g] + vecs[spk] * talk
            weights[g] += talk
        mappings.append(mapping)

    return mappings


def consolidate_speakers(
    window_embeddings: list[dict[str, tuple[np.ndarray, float]]],
    mappings: list[dict[str, str]],
    merge_threshold: float,
    center: bool = True,
    split_threshold: Optional[float] = None,
) -> list[dict[str, str]]:
    """Second pass: merge global speakers that are one person split in two.

    :func:`link_speakers` decides window by window from one window's worth of
    speech, and on a 76-min journal club it split the same speaker in two
    after a five-minute silence. Over the whole file the two voices had cosine
    similarity 0.84, the highest of any pair. But mean encoder embeddings are
    only moderately discriminative (distinct people reached 0.5-0.68), so a
    merge needs both conditions:

    * whole-file voice similarity of at least ``merge_threshold``, and
    * the two never speak in the same window — MOSS's within-window
      diarization is trusted to say two labels in one window are two people.

    Returns the mappings with merged labels rewritten, renumbered ``S01``,
    ``S02``, ... by first appearance.
    """
    sums: dict[str, np.ndarray] = {}
    talk: dict[str, float] = {}
    windows_of: dict[str, set[int]] = {}
    for i, (emb, mapping) in enumerate(zip(window_embeddings, mappings)):
        for local, (vec, secs) in emb.items():
            g = mapping.get(local)
            if g is None:
                continue
            sums[g] = sums.get(g, 0) + vec * secs
            talk[g] = talk.get(g, 0.0) + secs
            windows_of.setdefault(g, set()).add(i)
    if len(sums) < 2:
        return mappings

    labels = list(sums)
    common = np.mean([sums[g] / talk[g] for g in labels], axis=0) if center else 0.0
    unit = {g: _unit(sums[g] / talk[g] - common) for g in labels}
    parent = {g: g for g in labels}

    def root(g: str) -> str:
        while parent[g] != g:
            g = parent[g]
        return g

    pairs = sorted(
        ((float(np.dot(unit[a], unit[b])), a, b)
         for i, a in enumerate(labels) for b in labels[i + 1:]),
        reverse=True,
    )
    for sim, a, b in pairs:
        if sim < merge_threshold:
            break
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        members_a = [g for g in labels if root(g) == ra]
        members_b = [g for g in labels if root(g) == rb]
        seen_a = set().union(*(windows_of[g] for g in members_a))
        seen_b = set().union(*(windows_of[g] for g in members_b))
        if seen_a & seen_b and (split_threshold is None or sim < split_threshold):
            continue
        keep, drop = (ra, rb) if talk[ra] >= talk[rb] else (rb, ra)
        parent[drop] = keep

    # Renumber by first appearance so labels stay S01, S02, ... without gaps.
    order: dict[str, str] = {}
    for mapping in mappings:
        for g in mapping.values():
            r = root(g) if g in parent else g
            if r not in order:
                order[r] = f"S{len(order) + 1:02d}"
    return [
        {local: order[root(g) if g in parent else g] for local, g in mapping.items()}
        for mapping in mappings
    ]


def merge_windows(
    windows: list[Window],
    window_segments: list[list[dict]],
    mappings: list[dict[str, str]],
) -> list[dict]:
    """Stitch per-window segments into one global-time transcript.

    Each overlap is split at its midpoint; a segment belongs to whichever
    window its own midpoint falls in.
    """
    cuts = [
        (windows[i + 1].start + windows[i].end) / 2 for i in range(len(windows) - 1)
    ]
    merged: list[dict] = []
    for i, (win, segs, mapping) in enumerate(zip(windows, window_segments, mappings)):
        lo = cuts[i - 1] if i > 0 else -math.inf
        hi = cuts[i] if i < len(cuts) else math.inf
        for seg in segs:
            start, end = seg["start"] + win.start, seg["end"] + win.start
            if lo <= (start + end) / 2 < hi:
                merged.append({
                    **seg,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "speaker": mapping.get(seg["speaker"], seg["speaker"]),
                })
    merged.sort(key=lambda s: s["start"])
    return merged


# ── Window-level safeguards ──────────────────────────────────────────────────

# The pipeline levels transcription audio to about -20 dBFS RMS, so a 0.5 s
# frame quieter than -50 dBFS carries no speech worth decoding.
SILENCE_DBFS = -50.0
_FRAME_S = 0.5


def is_silent(audio: np.ndarray, sample_rate: int, min_active_s: float = 1.0) -> bool:
    """True when fewer than ``min_active_s`` seconds of the audio rise above
    :data:`SILENCE_DBFS`. Decoding silence wastes a batch slot and invites the
    model to invent text."""
    frame = int(_FRAME_S * sample_rate)
    n = len(audio) // frame
    if n == 0:
        return True
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    active = np.sum(20 * np.log10(rms + 1e-12) > SILENCE_DBFS) * _FRAME_S
    return active < min_active_s


def looks_degenerate(segments: list[dict], tokens: int, budget: int, repeats: int = 4) -> bool:
    """Signs a window's decode went wrong: it ran out its token budget, or it
    emitted the same multi-word segment ``repeats`` times in a row (a greedy
    decoding loop). Short backchannels like "Yeah." repeat legitimately."""
    if tokens >= budget:
        return True
    run, last = 0, None
    for seg in segments:
        text = " ".join(seg.get("text", "").lower().split())
        if len(text.split()) >= 3 and text == last:
            run += 1
            if run >= repeats - 1:
                return True
        else:
            run = 0
        last = text
    return False


# Speakers with less total talk than this across the whole file are flagged:
# some are real one-line interjections, some are a known voice mislabelled.
UNCERTAIN_TALK_S = 10.0


def flag_uncertain_speakers(segments: list[dict]) -> list[str]:
    """Mark segments whose speaker has little total talk; return those speakers."""
    talk: dict[str, float] = {}
    for seg in segments:
        talk[seg["speaker"]] = talk.get(seg["speaker"], 0.0) + seg["end"] - seg["start"]
    uncertain = sorted(k for k, v in talk.items() if v < UNCERTAIN_TALK_S)
    for seg in segments:
        if seg["speaker"] in uncertain:
            seg["speaker_uncertain"] = True
    return uncertain


def limit_speakers(
    segments: list[dict],
    embeddings: dict[str, tuple[np.ndarray, float]],
    n_speakers: int,
) -> list[dict]:
    """Fold a transcript down to a known number of speakers.

    For recordings where the count is known in advance (a two-person clinical
    interview), extra labels are MOSS or the linker splitting one person, not
    new people. The ``n_speakers`` labels with the most talk time are kept; each
    other label joins the kept speaker whose voice it matches best, or — with no
    usable voice sample — the kept speaker who talks most.

    ``embeddings`` maps label -> (voice embedding, talk seconds), as from
    ``VoiceEmbedder.speakers`` over the whole file.
    """
    talk: dict[str, float] = {}
    for seg in segments:
        talk[seg["speaker"]] = talk.get(seg["speaker"], 0.0) + max(seg["end"] - seg["start"], 0.0)
    if n_speakers < 1 or len(talk) <= n_speakers:
        return segments

    kept = sorted(talk, key=lambda s: -talk[s])[:n_speakers]
    target: dict[str, str] = {s: s for s in kept}
    for spk in talk:
        if spk in target:
            continue
        best, best_sim = kept[0], -2.0
        if spk in embeddings:
            v = _unit(embeddings[spk][0])
            for k in kept:
                if k in embeddings:
                    sim = float(v @ _unit(embeddings[k][0]))
                    if sim > best_sim:
                        best, best_sim = k, sim
        target[spk] = best
    return [{**seg, "speaker": target[seg["speaker"]]} for seg in segments]
