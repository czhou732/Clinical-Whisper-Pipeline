"""Measured elaboration by question valence."""

from elaboration import measure, question_valence

ROLES = {"I": "Interviewer", "S": "Subject"}


def seg(spk, text):
    return {"speaker": spk, "start": 0, "end": 1, "text": text}


def test_question_valence():
    assert question_valence("What do you enjoy doing on weekends?") == "positive"
    assert question_valence("What has been the hardest part of this year?") == "negative"
    assert question_valence("Where did you grow up?") == "neutral"
    assert question_valence("Okay, thank you.") is None
    assert question_valence("What do you enjoy, and what stresses you out?") == "neutral"


def test_words_per_answer_by_valence():
    segs = []
    for _ in range(3):
        segs += [seg("I", "What do you enjoy doing?"), seg("S", "Not much.")]
        segs += [seg("I", "Where do you live?"), seg("S", "I live in a small apartment near the river with my cat.")]
        segs += [seg("I", "What has been difficult lately?"), seg("S", "Everything has been hard, work and sleep and money.")]
    out = measure(segs, ROLES)
    assert out["positive"] == {"answers": 3, "words_mean": 2.0, "words_median": 2.0}
    assert out["neutral"]["words_median"] == 12.0
    assert out["positive_to_neutral"] == round(2 / 12, 2)


def test_answer_spans_several_subject_turns_and_stops_at_remarks():
    segs = [seg("I", "What do you love about your job?"), seg("S", "The people."),
            seg("S", "And the coffee."), seg("I", "Okay."), seg("S", "Yeah.")]
    out = measure(segs, ROLES)
    assert out["positive"]["words_mean"] == 5.0  # "Yeah" after a remark is not counted


def test_needs_an_interview():
    assert measure([seg("I", "Hi?")], {"I": "Moderator 1"}) is None
    out = measure([seg("I", "What do you enjoy?"), seg("S", "Music.")], ROLES)
    assert out["positive_to_neutral"] is None  # too few answers to compare
