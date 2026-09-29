"""Speech timing features from diarized segment timestamps.

These are deterministic: the same transcript always gives the same numbers,
so their test-retest reliability is limited only by transcription, unlike the
LLM-judged scores (ICC 0.22-0.60). They are also the measures with the most
direct literature in depression: slowed speech, longer and more variable
pauses, and delayed responses (psychomotor retardation). Pause-duration
variability was the top SHAP feature in the preprint's Stream A.

Definitions, per speaker:

* ``talk_time_s`` — summed segment duration.
* ``speech_rate_wps`` — words per second of talk time.
* ``turns`` — runs of consecutive segments by the same speaker.
* ``pause_*`` — gaps of at least :data:`MIN_PAUSE_S` between consecutive
  segments *within* one turn; ``pause_proportion`` is pause time over
  talk + pause time.
* ``response_latency_*`` — gap from the end of another speaker's segment to
  the start of this speaker's next turn (positive gaps only; overlaps are
  counted separately as ``overlap_starts``).
* ``filler_rate`` — filler words (um, uh, hmm, ...) per 100 words.
"""

from __future__ import annotations

import re
import statistics
from typing import Optional

MIN_PAUSE_S = 0.25
# A gap longer than this is a break in the conversation, not a pause or a response.
MAX_GAP_S = 10.0
FILLER = re.compile(r"\b(um+|uh+|erm|er|hmm+|mm+|mhm|mm-hmm|uh-huh)\b", re.I)


def _stats(values: list[float], prefix: str) -> dict:
    if not values:
        return {f"{prefix}_count": 0, f"{prefix}_mean_s": None,
                f"{prefix}_median_s": None, f"{prefix}_sd_s": None}
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean_s": round(statistics.fmean(values), 3),
        f"{prefix}_median_s": round(statistics.median(values), 3),
        f"{prefix}_sd_s": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def speaker_timing(segments: list[dict]) -> dict[str, dict]:
    """Timing features for every speaker in ``segments``."""
    segs = sorted(
        (s for s in segments if s.get("end", 0) > s.get("start", 0)),
        key=lambda s: s["start"],
    )
    out: dict[str, dict] = {}
    for spk in sorted({s["speaker"] for s in segs}):
        mine = [s for s in segs if s["speaker"] == spk]
        talk = sum(s["end"] - s["start"] for s in mine)
        words = sum(len(s.get("text", "").split()) for s in mine)
        fillers = sum(len(FILLER.findall(s.get("text", ""))) for s in mine)

        pauses: list[float] = []
        latencies: list[float] = []
        overlap_starts = 0
        turns = 0
        prev: Optional[dict] = None
        for s in segs:
            if s["speaker"] == spk:
                if prev is not None and prev["speaker"] == spk:
                    gap = s["start"] - prev["end"]
                    if MIN_PAUSE_S <= gap <= MAX_GAP_S:
                        pauses.append(gap)
                else:
                    turns += 1
                    if prev is not None:
                        gap = s["start"] - prev["end"]
                        if gap < 0:
                            overlap_starts += 1
                        elif gap <= MAX_GAP_S:
                            latencies.append(gap)
            prev = s

        pause_total = sum(pauses)
        out[spk] = {
            "talk_time_s": round(talk, 2),
            "words": words,
            "turns": turns,
            "speech_rate_wps": round(words / talk, 3) if talk > 0 else None,
            "pause_proportion": round(pause_total / (talk + pause_total), 4) if talk > 0 else None,
            **_stats(pauses, "pause"),
            **_stats(latencies, "response_latency"),
            "overlap_starts": overlap_starts,
            "filler_rate": round(100 * fillers / words, 2) if words else None,
        }
    return out


def subject_speaker(timing: dict[str, dict], roles: dict[str, str]) -> Optional[str]:
    """The speaker to report as the subject.

    Uses the role mapping when its keys are segment labels; otherwise the
    non-interviewer with the most talk time.
    """
    interviewers = {k for k, v in roles.items() if str(v).lower() in
                    ("interviewer", "clinician", "therapist", "ellie")}
    labelled = [k for k, v in roles.items() if str(v).lower() == "subject" and k in timing]
    if labelled:
        return labelled[0]
    candidates = [k for k in timing if k not in interviewers] or list(timing)
    return max(candidates, key=lambda k: timing[k]["talk_time_s"], default=None)
