"""Keyword screen for passages a clinician should read."""

import review_flags

ROLES = {"S01": "Interviewer", "S02": "Subject"}


def seg(spk, start, text):
    return {"speaker": spk, "start": start, "end": start + 5, "text": text}


def test_flags_participant_statements_with_the_question_as_context():
    out = review_flags.find([
        seg("S01", 0, "Have you had thoughts of suicide?"),
        seg("S02", 5, "Sometimes I feel like I want to die, yeah."),
    ], ROLES)
    assert [i["category"] for i in out["items"]] == ["suicide_self_harm"]
    item = out["items"][0]
    assert item["term"].lower() == "want to die"
    assert item["context"] == "Have you had thoughts of suicide?"
    assert item["start"] == 5
    assert out["note"] == review_flags.DISCLAIMER


def test_interviewer_questions_are_not_flagged():
    out = review_flags.find([seg("S01", 0, "Any thoughts of hurting yourself or suicide?")], ROLES)
    assert out["items"] == []


def test_negated_answers_are_still_shown():
    out = review_flags.find([seg("S02", 0, "No, I've never thought about suicide.")], ROLES)
    assert len(out["items"]) == 1


def test_whole_words_only():
    out = review_flags.find([seg("S02", 0, "The method was grape juice, then I drafted a therapist note.")],
                            ROLES)
    assert out["items"] == []


def test_empty_result_never_reads_as_all_clear():
    out = review_flags.find([seg("S02", 0, "I went to the park.")], ROLES)
    assert out["items"] == []
    assert "does not mean nothing concerning" in out["note"]


def test_extra_terms_from_config():
    out = review_flags.find([seg("S02", 0, "I've been taking xanax bars again.")], ROLES,
                            extra_terms={"substance": ["xanax"]})
    assert out["counts"] == {"substance": 1}
    assert out["items"][0]["label"] == "Substance use"


def test_group_moderators_are_context_not_flags():
    roles = {"S01": "Moderator 1", "S02": "Participant 1"}
    out = review_flags.find([seg("S01", 0, "Has anyone felt hopeless?"),
                             seg("S02", 5, "Honestly I felt hopeless last winter.")], roles)
    assert len(out["items"]) == 1 and out["items"][0]["role"] == "Participant 1"


def test_csv_summary():
    out = review_flags.find([seg("S02", 0, "I relapsed and felt worthless.")], ROLES)
    assert review_flags.summary(out) == "hopelessness:1; substance:1"
    assert review_flags.summary({}) == ""


def test_accidents_are_not_harm_to_others():
    out = review_flags.find([seg("S02", 0, "She face planted and hurt her nose.")], ROLES)
    assert out["items"] == []
    out = review_flags.find([seg("S02", 0, "Sometimes I want to hurt him.")], ROLES)
    assert out["counts"] == {"harm_to_others": 1}
