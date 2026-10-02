"""Scoring summaries that claim what the scorer can't know, or the measures contradict."""

import consistency


def test_voice_claims_are_flagged():
    flags = consistency.check("Low pitch variability suggests flat affect.", {})
    assert [f["code"] for f in flags] == ["summary_describes_voice"]


def test_fillers_claim_against_a_zero_rate():
    flags = consistency.check("The participant uses frequent fillers.", {"filler_rate": 0.0})
    assert "summary_contradicts_fillers" in [f["code"] for f in flags]
    assert consistency.check("The participant uses frequent fillers.", {"filler_rate": 6.0}) == []


def test_slow_speech_claim_against_fast_measured_rate():
    flags = consistency.check("Slow speech and long pauses throughout.",
                              {"speech_rate_wps": 3.1, "pause_mean_s": 0.4})
    assert "summary_contradicts_timing" in [f["code"] for f in flags]


def test_clean_summary_passes():
    assert consistency.check("The participant says they stopped playing guitar.", {}) == []
