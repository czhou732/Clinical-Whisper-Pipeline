"""Moving one transcript line to another speaker by hand."""

import pytest

import gui_server
from transcript_formatter import format_structured_transcript, move_turn, turn_spans

SEGS = [
    {"speaker": "S01", "start": 0.0, "end": 3.0, "text": "How was your week?"},
    {"speaker": "S02", "start": 3.5, "end": 6.0, "text": "Fine."},
    {"speaker": "S02", "start": 6.0, "end": 9.0, "text": "Busy, really."},
    {"speaker": "S01", "start": 9.5, "end": 10.0, "text": ""},
    {"speaker": "S01", "start": 10.0, "end": 12.0, "text": "And you, are you walking?"},
    {"speaker": "S02", "start": 75.0, "end": 79.0, "text": "I'm walking, sorry I'm late."},
]
ROLES = {"S01": "Moderator 1", "S02": "Participant 1"}


def test_turns_match_transcript_lines():
    lines = format_structured_transcript(SEGS, ROLES).splitlines()
    spans = turn_spans(SEGS)
    assert len(lines) == len(spans) == 4
    assert spans[1] == [1, 2] and spans[2] == [4]


def test_move_a_line_to_a_new_speaker():
    out, rec = move_turn(SEGS, 3, "new", start=75.0)
    assert out[5]["speaker"] == "S03" and out[5]["moved_by_hand"]
    assert rec == {"at": 75.0, "from": "S02", "to": "S03", "seconds": 4.0, "new_speaker": True}
    assert SEGS[5]["speaker"] == "S02"  # input untouched


def test_move_a_merged_line_moves_all_its_segments():
    out, rec = move_turn(SEGS, 1, "S01", start=3.0)
    assert [s["speaker"] for s in out[1:3]] == ["S01", "S01"] and rec["seconds"] == 5.5


def test_stale_or_wrong_requests_are_refused():
    with pytest.raises(ValueError, match="changed"):
        move_turn(SEGS, 3, "S01", start=12.0)
    with pytest.raises(ValueError, match="no longer"):
        move_turn(SEGS, 9, "S01")
    with pytest.raises(ValueError, match="already"):
        move_turn(SEGS, 0, "S01")
    with pytest.raises(ValueError, match="Unknown speaker"):
        move_turn(SEGS, 0, "S09")


def test_roles_after_move_add_a_participant_and_drop_empty_speakers():
    out, rec = move_turn(SEGS, 3, "new")
    roles = gui_server._roles_after_move(ROLES, out, rec)
    assert roles["S03"] == "Participant 2"
    only, rec2 = move_turn([SEGS[0], SEGS[5]], 1, "S01")
    assert gui_server._roles_after_move(ROLES, only, rec2) == {"S01": "Moderator 1"}


def test_interview_new_speaker_is_other():
    roles = {"S01": "Interviewer", "S02": "Subject"}
    out, rec = move_turn(SEGS, 3, "new")
    assert gui_server._roles_after_move(roles, out, rec)["S03"] == "Other_1"


def test_turn_owners_follow_lines():
    assert gui_server._turn_speakers(SEGS) == ["S01", "S02", "S01", "S02"]
