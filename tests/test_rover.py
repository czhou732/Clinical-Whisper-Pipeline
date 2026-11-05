"""Word-level voting across recognisers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals" / "ground_truth"))

from rover import combine, combine_segments  # noqa: E402


def test_majority_fixes_each_systems_own_error():
    ref = "the quick brown fox jumps over the lazy dog"
    a = "the quick brown box jumps over the lazy dog"
    b = "the quick brown fox jumped over the lazy dog"
    c = "the quack brown fox jumps over the lazy dog"
    assert combine([a, b, c]) == ref


def test_insertions_and_deletions_are_voted_too():
    a = "i went to the um store"
    b = "i went to the store"
    c = "i went to store"
    assert combine([a, b, c]) == "i went to the store"


def test_tie_goes_to_the_first_system():
    assert combine(["anthropic is here", "entropy is here"]) == "anthropic is here"


def test_single_and_empty():
    assert combine(["just one"]) == "just one"
    assert combine([]) == ""
    assert combine(["", "", "hello"]) == ""


def test_segments_keep_boundaries_and_speakers():
    moss = [{"speaker": "S01", "start": 0, "end": 2, "text": "how are you"},
            {"speaker": "S02", "start": 2, "end": 4, "text": "fine thanks"}]
    whisper = [{"text": "how are you"}, {"text": "find thanks"}]
    parakeet = [{"text": "who are you"}, {"text": "fine thanks"}]
    out = combine_segments([moss, whisper, parakeet])
    assert [s["text"] for s in out] == ["how are you", "fine thanks"]
    assert out[1]["speaker"] == "S02" and out[1]["start"] == 2
