"""Clinical scorer version 2: 0-3 scores that need a real quote."""

import json

import llm_clinical_scorer as s

TRANSCRIPT = """[00:00 - 00:03] Interviewer: What do you enjoy doing?
[00:03 - 00:09] Subject: Nothing really. I used to play guitar every night but I stopped.
[00:09 - 00:12] Interviewer: How is your mood?
[00:12 - 00:15] Subject: Pretty low most days."""


def answer(**overrides):
    base = {
        "anhedonia_content": {"score": 2, "evidence": ["I used to play guitar every night but I stopped"]},
        "depressed_mood_content": {"score": 2, "evidence": ["Pretty low most days"]},
        "affect_flatness": {"score": 1, "evidence": ["Nothing really"]},
        "engagement_level": {"score": 1, "evidence": ["What do you enjoy doing?"]},  # interviewer's words
        "summary": "The participant says they stopped playing guitar and feel low.",
    }
    base.update(overrides)
    return json.dumps(base)


def test_real_quotes_keep_scores_and_invented_ones_drop_them():
    parsed = s.check_evidence(s.parse_scoring_response(answer()), TRANSCRIPT)
    assert parsed["anhedonia_content"] == 2 and parsed["depressed_mood_content"] == 2
    assert parsed["engagement_level"] is None  # quoted the interviewer, not the participant
    invented = s.parse_scoring_response(answer(
        affect_flatness={"score": 3, "evidence": ["I feel completely numb inside"]}))
    assert s.check_evidence(invented, TRANSCRIPT)["affect_flatness"] is None


def test_out_of_range_and_null_scores_are_missing():
    parsed = s.parse_scoring_response(answer(anhedonia_content={"score": 7, "evidence": ["x"]},
                                             depressed_mood_content={"score": None, "evidence": []}))
    assert parsed["anhedonia_content"] is None and parsed["depressed_mood_content"] is None


def test_aggregate_needs_half_the_runs():
    runs = [{"anhedonia_content": 2, "depressed_mood_content": None, "affect_flatness": 1,
             "engagement_level": None, "evidence": {}, "summary": "a"},
            {"anhedonia_content": 3, "depressed_mood_content": None, "affect_flatness": None,
             "engagement_level": None, "evidence": {}, "summary": "longer"},
            {"anhedonia_content": 2, "depressed_mood_content": 1, "affect_flatness": None,
             "engagement_level": None, "evidence": {}, "summary": ""}]
    out = s._aggregate_scores(runs)
    assert out["anhedonia_content"] == 2.3
    assert out["depressed_mood_content"] is None   # 1 of 3 runs
    assert out["affect_flatness"] is None          # 1 of 3 runs
    assert out["summary"] == "longer"


def test_prompt_no_longer_carries_voice_numbers():
    assert "{acoustic_context}" not in s.CLINICAL_SCORING_PROMPT
    assert "cannot hear" in s.CLINICAL_SCORING_PROMPT
