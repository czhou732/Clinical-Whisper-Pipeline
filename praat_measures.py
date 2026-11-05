"""Praat voice measures, defined as in senselab (Bridge2AI-Voice).

senselab (sensein, Apache-2.0) is the extraction library behind the
Bridge2AI-Voice data. Its Praat settings are followed here exactly, so these
numbers can be compared with that dataset and with other labs that use it:

* pitch range per speaker: a wide autocorrelation search (50-600 Hz), outliers
  beyond 2 SD dropped, then 60-250 Hz below a 170 Hz mean, else 100-500 Hz
  (Vogel, Maruff & Morgan 2009);
* cross-correlation for period-level measures (jitter, shimmer, HNR) and
  autocorrelation for the pitch contour, as Praat recommends;
* CPPS on voiced stretches only, with senselab's cepstrogram and fit settings.

What this adds to OpenSMILE's eGeMAPS: CPPS (a standard clinical measure of
breathy or rough voice), HNR, several jitter and shimmer variants, and the
speaker-adaptive pitch range eGeMAPS lacks.

Praat's Python binding (praat-parselmouth) is GPL-3.0, and ClinicalWhisper is
MIT, so the binding is not bundled: it is a separate add-on (see addons.py).
Without it, these measures are simply absent.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("ClinicalWhisper")

SAMPLE_RATE = 16000
# Longest stretch of one speaker's speech analysed, sampled evenly across the
# session: CPPS walks every voiced stretch, and a long interview has thousands.
MAX_SPEECH_S = 300.0
MIN_SPEECH_S = 10.0


def available() -> bool:
    try:
        import addons
        addons.add_praat_path()
        import parselmouth  # noqa: F401
    except ImportError:
        return False
    return True


def _pitch_range(snd) -> tuple[float, float]:
    pitch = snd.to_pitch_ac(time_step=0.005, pitch_floor=50, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 != 0]
    if f0.size < 10:
        return 60.0, 250.0
    z = (f0 - f0.mean()) / (f0.std() or 1.0)
    mean = float(f0[np.abs(z) <= 2].mean())
    return (60.0, 250.0) if mean < 170 else (100.0, 500.0)


def _nan_to_none(d: dict) -> dict:
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
            for k, v in d.items()}


def measure(audio: np.ndarray) -> dict:
    """Praat measures for one speaker's speech (16 kHz mono float32)."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    floor, ceiling = _pitch_range(snd)
    out: dict = {"pitch_floor_hz": floor, "pitch_ceiling_hz": ceiling}

    pitch = snd.to_pitch_ac(time_step=0.005, pitch_floor=floor, pitch_ceiling=ceiling)
    out["f0_mean_hz"] = call(pitch, "Get mean", 0, 0, "Hertz")
    out["f0_sd_hz"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=floor, silence_threshold=0.1,
                                 periods_per_window=4.5)
    out["hnr_db_mean"] = call(harm, "Get mean", 0, 0)
    out["hnr_db_sd"] = call(harm, "Get standard deviation", 0, 0)

    pp = call(snd, "To PointProcess (periodic, cc)", floor, ceiling)
    for name, kind in (("jitter_local", "local"), ("jitter_rap", "rap"), ("jitter_ppq5", "ppq5")):
        out[name] = call(pp, f"Get jitter ({kind})", 0, 0, 0.0001, 0.02, 1.3)
    for name, kind in (("shimmer_local", "local"), ("shimmer_local_db", "local_dB"),
                       ("shimmer_apq3", "apq3"), ("shimmer_apq5", "apq5")):
        out[name] = call([snd, pp], f"Get shimmer ({kind})", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    out.update(_cpps(snd, floor, ceiling))
    ltas = call(snd, "To Ltas", 100)
    out["spectral_slope_db"] = call(ltas, "Get slope", 50, 1000, 1000, 4000, "dB")
    return _nan_to_none(out)


def _cpps(snd, floor: float, ceiling: float) -> dict:
    from parselmouth.praat import call

    pitch = snd.to_pitch_ac(time_step=0.005, pitch_floor=floor, pitch_ceiling=ceiling,
                            voicing_threshold=0.3)
    pulses = call([snd, pitch], "To PointProcess (cc)")
    grid = call(pulses, "To TextGrid (vuv)", 0.02, 0.1)
    table = call(grid, "Down to Table", "no", 6, "yes", "no")
    values = []
    for i in range(int(call(table, "Get number of rows"))):
        if call(table, "Get value", i + 1, "text") != "V":
            continue
        tmin = float(call(table, "Get value", i + 1, "tmin"))
        tmax = float(call(table, "Get value", i + 1, "tmax"))
        if tmax - tmin < 0.05:
            continue
        part = snd.extract_part(tmin, tmax)
        try:
            ceps = call(part, "To PowerCepstrogram", 60, 0.002, 5000, 50)
            v = call(ceps, "Get CPPS...", "no", 0.01, 0.001, 60, 330, 0.05, "parabolic",
                     0.001, 0, "Straight", "Robust")
        except Exception:  # noqa: BLE001 - one bad stretch must not lose the rest
            continue
        if np.isfinite(v) and v > 4:  # senselab's cut-off for implausible values
            values.append(v)
    return {"cpps_mean": float(np.mean(values)) if values else None,
            "cpps_sd": float(np.std(values)) if values else None}


def speaker_audio(wav_path: str, segments: list[dict], speaker: str,
                  max_s: float = MAX_SPEECH_S) -> np.ndarray:
    """Up to ``max_s`` of one speaker's speech, taken evenly across the session."""
    import soundfile as sf

    mine = [s for s in segments if s.get("speaker") == speaker and s["end"] - s["start"] >= 0.5]
    total = sum(s["end"] - s["start"] for s in mine)
    if total > max_s:
        step = total / max_s
        kept, acc, nxt = [], 0.0, 0.0
        for s in mine:  # every step-th second of speech, segment by segment
            if acc >= nxt:
                kept.append(s)
                nxt += (s["end"] - s["start"]) * step
            acc += s["end"] - s["start"]
        mine = kept
    pieces = []
    with sf.SoundFile(wav_path) as f:
        for s in mine:
            f.seek(int(s["start"] * SAMPLE_RATE))
            pieces.append(f.read(int((s["end"] - s["start"]) * SAMPLE_RATE), dtype="float32"))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def per_speaker(wav_path: str, segments: list[dict]) -> Optional[dict]:
    """Praat measures for every speaker with enough speech, or None without the add-on."""
    if not available():
        return None
    talk: dict[str, float] = {}
    for s in segments:
        talk[s.get("speaker")] = talk.get(s.get("speaker"), 0.0) + max(0.0, s["end"] - s["start"])
    out = {}
    for spk, secs in talk.items():
        if secs < MIN_SPEECH_S:
            continue
        try:
            audio = speaker_audio(wav_path, segments, spk)
            out[spk] = {**measure(audio), "speech_s_analysed": round(len(audio) / SAMPLE_RATE, 1)}
        except Exception as exc:  # noqa: BLE001 - optional measures never fail a run
            log.warning("Praat measures failed for %s: %s", spk, exc)
    return out
