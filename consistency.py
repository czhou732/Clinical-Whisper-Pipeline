"""Catch a scoring summary that claims what the scorer cannot know or the measures contradict.

The scorer reads a transcript. It cannot hear pitch, voice quality or pauses,
yet version 1 wrote "agitation or dysarthria" from jitter numbers and
"frequent fillers" when the measured filler rate was zero. Version 2's prompt
forbids voice comments; this checks the output anyway and flags any line that
breaks the rule or disagrees with a measurement taken from the audio.
"""

from __future__ import annotations

import re
from typing import Optional

_VOICE = re.compile(r"\b(?:pitch|jitter|shimmer|prosod\w*|monoton\w*|dysarthri\w*|slurr\w*|"
                    r"tone of voice|vocal\w*|voice quality|loudness|volume)\b", re.I)
_FILLERS = re.compile(r"\b(?:fillers?|um+s?|uhs?|hesitat\w*)\b", re.I)
_SLOW = re.compile(r"\b(?:slow(?:ed|ly|ness)? (?:speech|rate|speaking|response)s?|"
                   r"long pauses|psychomotor (?:slowing|retardation))\b", re.I)


def check(text: str, timing: Optional[dict]) -> list[dict]:
    """Quality flags for claims in ``text`` (summary and observations)."""
    flags = []
    if not text:
        return flags
    if _VOICE.search(text):
        flags.append({"code": "summary_describes_voice", "message": (
            "The scoring summary comments on the voice, which the scorer cannot hear (it "
            "reads only the transcript). Use the measured voice values instead.")})
    t = timing or {}
    fillers = t.get("filler_rate")
    if _FILLERS.search(text) and isinstance(fillers, (int, float)) and fillers < 1.0:
        flags.append({"code": "summary_contradicts_fillers", "message": (
            f"The scoring summary mentions fillers or hesitation, but the measured rate is "
            f"{fillers:.1f} per 100 words.")})
    rate, pause = t.get("speech_rate_wps"), t.get("pause_mean_s")
    if _SLOW.search(text) and isinstance(rate, (int, float)) and rate >= 2.5 and \
            (not isinstance(pause, (int, float)) or pause < 0.8):
        flags.append({"code": "summary_contradicts_timing", "message": (
            f"The scoring summary describes slow speech or long pauses, but the measured "
            f"speech rate is {rate:.1f} words/s.")})
    return flags
