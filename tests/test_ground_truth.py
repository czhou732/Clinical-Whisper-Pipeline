"""Sanity checks for the ground-truth scorer: known inputs, known errors."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "ground_truth"))

from score import cpwer, der, filler_precision_recall, normalize, score, wer  # noqa: E402

REF = [
    {"speaker": "A", "start": 0.0, "end": 4.0, "text": "so um we tested the model"},
    {"speaker": "B", "start": 4.5, "end": 7.0, "text": "mm-hmm and it worked"},
    {"speaker": "A", "start": 7.5, "end": 10.0, "text": "uh mostly yes"},
]


def test_reference_against_itself_scores_perfectly():
    s = score(REF, REF)
    assert s["cpwer"] == 0 and s["wer_verbatim"] == 0 and s["der"] == 0
    assert s["filler_precision"] == 1 and s["filler_recall"] == 1


def test_renamed_speakers_are_not_errors():
    renamed = [{**seg, "speaker": {"A": "S02", "B": "S01"}[seg["speaker"]]} for seg in REF]
    assert cpwer(REF, renamed) == 0
    assert der(REF, renamed)["der"] == 0


def test_words_right_but_speaker_wrong_counts_in_cpwer_not_wer():
    wrong = [dict(seg) for seg in REF]
    wrong[2]["speaker"] = "B"  # A's last turn attributed to B
    assert wer(REF, wrong) == 0
    assert cpwer(REF, wrong) > 0
    assert der(REF, wrong)["confusion"] > 0


def test_one_deleted_word_is_one_error():
    n_words = len(normalize(" ".join(s["text"] for s in REF)))
    dropped = [dict(seg) for seg in REF]
    dropped[0]["text"] = "so um we tested model"
    assert wer(REF, dropped) == pytest.approx(1 / n_words)


def test_filler_spelling_variants_match():
    assert normalize("Mm-hmm, umm... Hmm") == normalize("MHM UM MM")
    assert normalize("180") == "one hundred eighty".split()


def test_dropped_filler_lowers_recall_not_precision():
    no_um = [dict(seg) for seg in REF]
    no_um[0]["text"] = "so we tested the model"
    precision, recall = filler_precision_recall(REF, no_um)
    assert precision == 1 and recall == pytest.approx(2 / 3)


def test_missing_segment_is_missed_speech():
    missing = REF[:2]
    d = der(REF, missing)
    assert d["missed"] > 0 and d["false_alarm"] == 0 and d["confusion"] == 0
