"""What a result can and cannot be trusted for, attached to every analysis.

Four small pieces, all about being honest in a clinical setting:

* :data:`RESEARCH_USE_NOTICE`: the intended-use statement written into every
  output. ClinicalWhisper is a research instrument, not a medical device.
* :func:`score_reliability`: the measured test-retest reliability of each LLM
  clinical score, so a number is never shown without how much it moves on
  re-scoring. Source: ``evals/reports/reliability.md`` (9 clips from separate
  recordings, 5 runs each, temperature 0.7; ICC(1,1), Shrout & Fleiss 1979).
  The CIs there are wide (n = 9); these are point estimates.
* :func:`assess_quality`: per-recording flags for input that makes the numbers
  unreliable (little participant speech, clipping, very quiet audio, brief or
  unexpected speakers).
* :func:`filevault_on`: whether the disk the outputs are written to is
  encrypted. The app does not encrypt files itself; FileVault is what protects
  transcripts and audio at rest.
"""

from __future__ import annotations

import math
import subprocess
from typing import Optional

RESEARCH_USE_NOTICE = (
    "Research use only. ClinicalWhisper is not a medical device: it does not "
    "diagnose, screen for, or monitor any condition, and its scores are "
    "exploratory research measures, not clinical assessments."
)

# ICC(1,1) of a single scoring run, from evals/reports/reliability.md.
_ICC_SINGLE_RUN = {
    "hesitancy_score": 0.288,
    "affect_flatness": 0.501,
    "engagement_level": 0.366,
    "elaboration_positive": 0.596,
    "elaboration_negative": 0.533,
    "psychomotor_indicators": 0.217,
}
# "Good" reliability starts at 0.75 (Koo & Li, 2016).
ADEQUATE_ICC = 0.75
# Two LLM scores ask for timing the model cannot see in a text transcript. The
# pipeline measures that timing directly, so these columns are the ones to use.
_MEASURED_INSTEAD = {
    "hesitancy_score": ["subject_pause_mean_s", "subject_pause_proportion", "subject_filler_rate"],
    "psychomotor_indicators": ["subject_response_latency_median_s", "subject_speech_rate_wps"],
}


def spearman_brown(icc: float, runs: int) -> float:
    """Reliability of the mean of ``runs`` independent scorings."""
    return runs * icc / (1 + (runs - 1) * icc)


def score_reliability(runs: int) -> dict[str, dict]:
    """Per score: expected ICC for the mean of ``runs`` scorings, and what it means."""
    runs = max(1, int(runs))
    out = {}
    for key, icc1 in _ICC_SINGLE_RUN.items():
        icc = round(spearman_brown(icc1, runs), 2)
        entry = {"icc": icc, "runs": runs, "adequate": icc >= ADEQUATE_ICC}
        if key in _MEASURED_INSTEAD:
            entry["use_instead"] = _MEASURED_INSTEAD[key]
        out[key] = entry
    return out


# --- Recording quality -------------------------------------------------------

# Working thresholds, not validated cut-offs: below them the measures are
# computed from too little signal to compare across people.
MIN_RECORDING_S = 60.0
MIN_PARTICIPANT_SPEECH_S = 180.0
MAX_CLIPPED_FRACTION = 0.001
QUIET_DBFS = -40.0


def _flag(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def assess_quality(
    segments: list[dict],
    subject_speaker: Optional[str],
    audio_stats: Optional[dict],
    expected_speakers: Optional[int] = None,
) -> dict:
    """Flags for recordings whose measures should be read with caution."""
    talk: dict[str, float] = {}
    for seg in segments:
        talk[seg.get("speaker")] = talk.get(seg.get("speaker"), 0.0) + max(
            seg.get("end", 0.0) - seg.get("start", 0.0), 0.0)
    participant_s = talk.get(subject_speaker, 0.0) if subject_speaker else 0.0
    stats = audio_stats or {}
    duration = stats.get("duration_s")
    flags = []

    if duration is not None and duration < MIN_RECORDING_S:
        flags.append(_flag("short_recording",
                           f"The recording is {duration:.0f} s long; most measures need "
                           "several minutes of conversation."))
    if not subject_speaker:
        flags.append(_flag("no_participant",
                           "No participant was identified, so participant measures are empty."))
    elif participant_s < MIN_PARTICIPANT_SPEECH_S:
        flags.append(_flag("little_participant_speech",
                           f"Only {participant_s / 60:.1f} min of participant speech. Timing "
                           "and voice measures from under 3 min are noisy."))
    clipped = stats.get("clipped_fraction")
    if clipped is not None and clipped > MAX_CLIPPED_FRACTION:
        flags.append(_flag("clipping",
                           f"{clipped:.2%} of samples are clipped (recorded too loud); "
                           "jitter, shimmer and loudness are distorted."))
    level = stats.get("rms_dbfs")
    if level is not None and level < QUIET_DBFS:
        flags.append(_flag("quiet_recording",
                           f"The recording is very quiet ({level:.0f} dBFS); words and "
                           "pauses may be missed."))
    brief = sorted({s["speaker"] for s in segments if s.get("speaker_uncertain")})
    if brief:
        flags.append(_flag("brief_speakers",
                           f"{len(brief)} speaker label(s) with under 10 s of talk "
                           f"({', '.join(brief)}); they may be one person split in two."))
    if expected_speakers and len(talk) != expected_speakers:
        flags.append(_flag("speaker_count",
                           f"{len(talk)} speakers found; {expected_speakers} expected."))

    return {
        "flags": flags,
        "participant_speech_s": round(participant_s, 1),
        "duration_s": None if duration is None else round(duration, 1),
        "rms_dbfs": None if level is None else round(level, 1),
        "clipped_fraction": clipped,
    }


def level_stats(sum_squares: float, count: int, clipped: int, sample_rate: int) -> dict:
    """Whole-file level and clipping from the decoder's running totals."""
    if not count:
        return {"duration_s": 0.0, "rms_dbfs": None, "clipped_fraction": None}
    rms = math.sqrt(sum_squares / count)
    return {
        "duration_s": count / sample_rate,
        "rms_dbfs": 20 * math.log10(rms) if rms > 0 else -120.0,
        "clipped_fraction": clipped / count,
    }


# --- Storage -----------------------------------------------------------------

def filevault_on() -> Optional[bool]:
    """True/False for macOS disk encryption; None where it cannot be told."""
    try:
        out = subprocess.run(["fdesetup", "status"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "FileVault is On" in out:
        return True
    if "FileVault is Off" in out:
        return False
    return None
