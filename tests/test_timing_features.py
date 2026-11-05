"""Timing features on hand-built segments with hand-computed answers."""

import pytest

from timing_features import speaker_timing, subject_speaker

SEGS = [
    {"speaker": "I", "start": 0.0, "end": 2.0, "text": "how are you feeling"},
    # Subject answers 1.5 s later, pauses 0.5 s mid-turn.
    {"speaker": "S", "start": 3.5, "end": 5.5, "text": "um I guess okay"},
    {"speaker": "S", "start": 6.0, "end": 8.0, "text": "not great really"},
    {"speaker": "I", "start": 8.2, "end": 9.0, "text": "tell me more"},
    # Subject starts before the interviewer finishes: an overlap, not a latency.
    {"speaker": "S", "start": 8.8, "end": 10.8, "text": "uh it has been hard"},
]


def test_subject_timing():
    t = speaker_timing(SEGS)["S"]
    assert t["talk_time_s"] == 6.0
    assert t["words"] == 12
    assert t["turns"] == 2
    assert t["speech_rate_wps"] == pytest.approx(12 / 6, abs=1e-3)
    assert t["pause_count"] == 1 and t["pause_mean_s"] == 0.5
    assert t["pause_proportion"] == pytest.approx(0.5 / 6.5, abs=1e-4)
    assert t["response_latency_count"] == 1 and t["response_latency_mean_s"] == 1.5
    assert t["overlap_starts"] == 1
    assert t["filler_rate"] == pytest.approx(100 * 2 / 12, abs=0.01)


def test_long_gaps_are_breaks_not_pauses():
    segs = [
        {"speaker": "S", "start": 0, "end": 2, "text": "one"},
        {"speaker": "S", "start": 30, "end": 32, "text": "two"},
    ]
    assert speaker_timing(segs)["S"]["pause_count"] == 0


def test_subject_choice():
    timing = speaker_timing(SEGS)
    assert subject_speaker(timing, {"S": "subject", "I": "interviewer"}) == "S"
    # Role keys that are display names, not segment labels: fall back to talk time.
    assert subject_speaker(timing, {"Speaker 1": "interviewer"}) == "S"
    assert subject_speaker(timing, {"I": "interviewer"}) == "S"
