"""Clinical scorer version 2: 0-3 scores that need a real quote."""

import json
import sys

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


def test_cli_model_override_reaches_both_local_backends(tmp_path, monkeypatch, capsys):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(TRANSCRIPT, encoding="utf-8")
    seen = {}

    def fake_score(structured_transcript, acoustic_context, config):
        seen.update(config["llm_scoring"])
        return {"_meta": {"model": "custom/model"}}

    monkeypatch.setattr(s, "score_transcript", fake_score)
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm_clinical_scorer.py", "--file", str(transcript),
         "--model", "custom/model", "--json"],
    )

    s.main()

    assert seen["mlx_model"] == "custom/model"
    assert seen["hf_model"] == "custom/model"
    assert json.loads(capsys.readouterr().out)["_meta"]["model"] == "custom/model"


def test_human_readable_output_uses_v2_scales():
    rendered = s._format_scoring_output({
        "anhedonia_content": 2,
        "depressed_mood_content": 1,
        "affect_flatness": None,
        "engagement_level": 3,
        "summary": "Research summary.",
    })

    assert "Interest and Pleasure" in rendered
    assert "2/3" in rendered
    assert "Hesitancy" not in rendered
    assert "/10" not in rendered


# What the app actually scores: identifiers masked as [tag_n], and a named
# speaker shown as "Name (Subject)". Synthetic text, no real person.
MASKED = """[00:00 - 00:05] Interviewer: Where did you grow up?
[00:05 - 00:20] Subject: I grew up in [city_1]. My brother [first_name_1] moved to [city_2] a few years ago.
[00:20 - 00:23] Interviewer: What do you enjoy doing on the weekends?
[00:23 - 00:25] Subject: Not much, honestly.
[00:25 - 00:28] Interviewer: How has your mood been?
[00:28 - 00:50] Subject: Pretty low. Most days I feel heavy, like nothing is going to get better. I lie there going over everything that went wrong with [first_name_1].
[00:50 - 00:52] Interviewer: What has been the hardest part?
[00:52 - 01:10] Sam (Subject): I played guitar every night and went running with [first_name_2]. I stopped both, and I do not even miss them. I do not really watch TV either, it is just noise."""

MASKED_ANSWER = json.dumps({
    "anhedonia_content": {"score": 3, "evidence": ["I stopped both, and I do not even miss them"]},
    "depressed_mood_content": {"score": 3, "evidence": ["Most days I feel heavy, like nothing is going to get better."]},
    "affect_flatness": {"score": 2, "evidence": ["it is just noise"]},
    "engagement_level": {"score": 2, "evidence": ["My brother [first_name_1] moved to [city_2] a few years ago"]},
    "summary": "The participant describes low mood and loss of interest.",
})


def test_quotes_from_lines_with_masked_identifiers_count():
    # The 5.3.0 bug: a participant line with a masking tag was left out of the
    # words quotes are checked against, so its quotes were "invented".
    said = s.participant_text(MASKED)
    assert "pretty low" in said and "i stopped both" in said and "first_name_1" in said
    assert "where did you grow up" not in said        # the interviewer's words stay out
    assert "00 23" not in said and "subject" not in said  # no timestamps or labels
    parsed = s.check_evidence(s.parse_scoring_response(MASKED_ANSWER), MASKED)
    assert {k: parsed[k] for k in s.REQUIRED_SCORE_KEYS} == {
        "anhedonia_content": 3, "depressed_mood_content": 3,
        "affect_flatness": 2, "engagement_level": 2}


def test_masked_anhedonic_interview_scores_come_back(monkeypatch):
    # End to end through score_transcript with a stand-in model, 5 runs as in
    # the app: every score must be reported, none missing.
    monkeypatch.setattr(s, "load", None)  # one pass at a time, no MLX
    monkeypatch.setattr(s, "call_local_lm", lambda **kw: MASKED_ANSWER)
    result = s.score_transcript(MASKED, config={"llm_scoring": {"samples": 5, "temperature": 0.7}})
    assert result["_meta"]["runs_used"] == 5
    assert {k: result[k] for k in s.REQUIRED_SCORE_KEYS} == {
        "anhedonia_content": 3.0, "depressed_mood_content": 3.0,
        "affect_flatness": 2.0, "engagement_level": 2.0}
