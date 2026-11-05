"""Prompt-response pairs: the unit both mechanism models are fit to.

A *response* is the participant's speech after an interviewer prompt, until
the interviewer speaks again. Operational definitions:

* ``latency_s`` — prompt end to the participant's first segment start. Pairs
  with overlap (participant starts before the prompt ends) or a gap longer
  than :data:`MAX_LATENCY_S` are kept but flagged, and excluded from latency
  models: the first is barge-in, the second an interruption or dead air.
* ``pauses_s`` — gaps of at least :data:`MIN_PAUSE_S` between consecutive
  participant segments within the response.
* ``speech_s``, ``words``, ``speech_rate_wps`` — summed segment time, word
  count, and their ratio.
* ``fillers`` — filler tokens (um, uh, hmm, ...).
* ``valence`` — the prompt's label from a rater-made map (positive / neutral /
  negative), or ``None`` if the prompt is not in the map.
"""

from __future__ import annotations

import re
from typing import Optional

from research.speech_mechanisms.transcripts import INTERVIEWER, PARTICIPANT, normalize_prompt

MIN_PAUSE_S = 0.25
MAX_LATENCY_S = 10.0
FILLER = re.compile(r"\b(um+|uh+|erm|hmm+|mm+|mhm|mm-hmm|uh-huh)\b", re.I)


def prompt_valence(prompt: str, valence_map: dict[str, str]) -> Optional[str]:
    """Label for the first map pattern contained in the normalized prompt."""
    text = normalize_prompt(prompt)
    for pattern, label in valence_map.items():
        if normalize_prompt(pattern) in text:
            return label
    return None


def pairs(segments: list[dict], valence_map: Optional[dict[str, str]] = None) -> list[dict]:
    valence_map = valence_map or {}
    out = []
    for i, seg in enumerate(segments):
        if seg["role"] != INTERVIEWER:
            continue
        response = []
        for nxt in segments[i + 1:]:
            if nxt["role"] == INTERVIEWER:
                break
            if nxt["role"] == PARTICIPANT:
                response.append(nxt)
        if not response:
            continue
        latency = response[0]["start"] - seg["end"]
        pauses = [
            b["start"] - a["end"] for a, b in zip(response, response[1:])
            if b["start"] - a["end"] >= MIN_PAUSE_S
        ]
        speech = sum(r["end"] - r["start"] for r in response)
        words = sum(len(r["text"].split()) for r in response)
        out.append({
            "prompt": seg["text"],
            "valence": prompt_valence(seg["text"], valence_map),
            "latency_s": latency,
            "latency_valid": 0 <= latency <= MAX_LATENCY_S,
            "pauses_s": pauses,
            "speech_s": speech,
            "words": words,
            "speech_rate_wps": words / speech if speech > 0 else None,
            "fillers": sum(len(FILLER.findall(r["text"])) for r in response),
        })
    return out
