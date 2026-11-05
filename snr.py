"""Signal-to-noise ratio of a recording, from its own speech and silences.

Background noise (air conditioning, a hallway, a distant microphone) inflates
jitter and shimmer and hides short pauses, so a noisy recording's voice
measures should be read with caution. This estimates how far the voice sits
above the noise floor:

* speech level: the average energy of 30 ms frames inside speaker segments;
* noise floor: the 10th percentile of frames outside every segment, when the
  recording has at least 20 s of such gaps; otherwise the 5th percentile of
  all frames (the quietest moments are then short pauses within speech).

It is an estimate, good to a few dB: enough to separate a clean interview
(25-40 dB) from a noisy one (under 15 dB), not a lab measurement.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.03
# Frames this close to a segment edge are ignored: segment boundaries are
# approximate, and a frame there may hold either speech or silence.
EDGE_S = 0.2
MIN_GAP_S = 20.0
NOISY_DB = 15.0
MAX_DB = 60.0
_BLOCK_S = 60.0


def _frame_db(path: str) -> tuple[np.ndarray, float]:
    """Loudness (dBFS) of every 30 ms frame, read a minute at a time."""
    import soundfile as sf

    out = []
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        n = int(FRAME_S * sr)
        for block in f.blocks(blocksize=int(_BLOCK_S * sr) // n * n, dtype="float32",
                              always_2d=True):
            mono = block.mean(axis=1)
            usable = len(mono) // n * n
            if not usable:
                continue
            frames = mono[:usable].reshape(-1, n)
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            out.append(20 * np.log10(np.maximum(rms, 1e-10)))
    return (np.concatenate(out) if out else np.empty(0)), FRAME_S


def estimate_snr(path: str, segments: list[dict]) -> float | None:
    """Speech level minus noise floor in dB, or None if it can't be measured."""
    try:
        db, step = _frame_db(path)
    except (OSError, RuntimeError, ValueError):
        return None
    if len(db) < 100 or not segments:
        return None
    inside = np.zeros(len(db), dtype=bool)
    near = np.zeros(len(db), dtype=bool)

    def _idx(seconds: float) -> int:
        return min(len(db), max(0, round(seconds / step)))

    for seg in segments:
        start, end = seg.get("start", 0.0), seg.get("end", 0.0)
        inside[_idx(start + EDGE_S):_idx(end - EDGE_S)] = True
        near[_idx(start - EDGE_S):_idx(end + EDGE_S)] = True
    gaps = ~near
    if not inside.any():
        return None
    # Energy average, the usual definition of a speech level.
    speech = float(10 * np.log10(np.mean(10 ** (db[inside] / 10))))
    if gaps.sum() * step >= MIN_GAP_S:
        noise = float(np.percentile(db[gaps], 10))
    else:
        noise = float(np.percentile(db, 5))
    # Above 60 dB is "clean" for every purpose here; larger values only mean
    # the gaps are digital silence (a synthetic or edited file).
    return round(min(speech - noise, MAX_DB), 1)
