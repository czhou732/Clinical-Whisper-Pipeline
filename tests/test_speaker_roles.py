"""Automatic roles: interviews, groups, uncertainty, guide and known voices."""

import numpy as np

import moss_chunking
from speaker_roles import assign
import voice_library


def seg(spk, start, end, text):
    return {"speaker": spk, "start": start, "end": end, "text": text}


def interview():
    out, t = [], 0.0
    for i in range(12):
        out.append(seg("A", t, t + 4, "How have you been sleeping? What about your appetite?"))
        out.append(seg("B", t + 4, t + 34, "Not great honestly. I wake up early and I can't get back "
                       "to sleep, and then the whole day feels heavy and slow."))
        t += 34
    return out


def test_interviewer_is_the_one_asking_questions():
    r = assign(interview())
    assert r["mode"] == "interview" and not r["uncertain"]
    assert r["roles"] == {"A": "Interviewer", "B": "Subject"}


def test_group_moderators_and_participants():
    segs, t = [], 0.0
    segs.append(seg("M1", t, t + 20, "Welcome, thank you for joining. Your participation is "
                    "completely voluntary and you may skip any question."))
    t = 20
    for i in range(8):
        segs.append(seg("M1", t, t + 6, "What about you, [first_name_1]? How do you get around?"))
        segs.append(seg("P1", t + 6, t + 46, "I mostly use my cane and an app on my phone to read things."))
        segs.append(seg("M2", t + 46, t + 52, "And you? What do you think about the glasses?"))
        segs.append(seg("P2", t + 52, t + 92, "I like the idea but I worry about privacy in public places."))
        t += 92
    r = assign(segs)
    assert r["mode"] == "group" and not r["uncertain"]
    assert {r["roles"]["M1"], r["roles"]["M2"]} == {"Moderator 1", "Moderator 2"}
    assert {r["roles"]["P1"], r["roles"]["P2"]} == {"Participant 1", "Participant 2"}


def test_two_people_chatting_is_uncertain():
    segs, t = [], 0.0
    for i in range(10):
        segs.append(seg("A", t, t + 20, "I think we should change the protocol. What do you think?"))
        segs.append(seg("B", t + 20, t + 40, "Maybe, but the survey is done. Should we email them?"))
        t += 40
    r = assign(segs)
    assert r["uncertain"] and r["why"]


def test_guide_questions_identify_the_interviewer():
    segs, t = [], 0.0
    for i in range(10):
        segs.append(seg("A", t, t + 20, "Over the past month have you lost interest in things you "
                        "used to enjoy? I wonder."))
        segs.append(seg("B", t + 20, t + 40, "Yes, I wonder about that too. What do you mean exactly?"))
        t += 40
    guide = "1. Over the past month have you lost interest in things you used to enjoy?\n"
    r = assign(segs, guide_text=guide)
    assert r["roles"]["A"] == "Interviewer" and not r["uncertain"]
    assert any("guide" in x for x in r["evidence"]["A"]["reasons"])


def test_known_voice_decides(tmp_path):
    lib = tmp_path / "voices.json"
    voice_library.remember("Moderator A", "Interviewer", np.array([1.0, 0.0, 0.0]), path=lib)
    segs, t = [], 0.0
    for i in range(10):
        segs.append(seg("X", t, t + 20, "So then I went to the store and it was fine."))
        segs.append(seg("Y", t + 20, t + 40, "Right, and then what happened after the store?"))
        t += 40
    voices = {"X": np.array([0.99, 0.05, 0.0]), "Y": np.array([0.0, 1.0, 0.0])}
    r = assign(segs, voices=voices, library=voice_library.load(lib))
    assert r["roles"]["X"] == "Interviewer"
    assert "voice matches Moderator A" in r["evidence"]["X"]["reasons"]


def test_voice_library_only_takes_staff(tmp_path):
    lib = tmp_path / "voices.json"
    import pytest
    with pytest.raises(ValueError):
        voice_library.remember("P01", "Subject", np.ones(3), path=lib)
    voice_library.remember("Moderator A", "Moderator 1", np.ones(3), path=lib)
    voice_library.remember("Moderator A", "Moderator 1", np.ones(3), path=lib)
    assert voice_library.load(lib)[0]["sessions"] == 2
    assert voice_library.forget("Moderator A", path=lib) and voice_library.labels(lib) == []


def test_late_unmatched_voice_is_kept_separate():
    segs = [seg("S1", 0, 300, "a"), seg("S2", 300, 600, "b"), seg("S3", 2000, 2010, "late")]
    emb = {"S1": (np.array([1.0, 0, 0]), 300), "S2": (np.array([0, 1.0, 0]), 300),
           "S3": (np.array([0, 0, 1.0]), 10)}
    out = moss_chunking.limit_speakers(segs, emb, 2, keep_new_below=0.5)
    assert {s["speaker"] for s in out} == {"S1", "S2", "S3"}
    # Without the rule, or for an early fragment, it is folded as before.
    assert {s["speaker"] for s in moss_chunking.limit_speakers(segs, emb, 2)} == {"S1", "S2"}
    early = [seg("S1", 0, 300, "a"), seg("S2", 300, 600, "b"), seg("S3", 30, 40, "early")]
    assert {s["speaker"] for s in moss_chunking.limit_speakers(early, emb, 2, keep_new_below=0.5)} == {"S1", "S2"}


def test_short_group_with_brief_moderators():
    """A one-minute group: moderators talk 12-17 s, participants 21 s each."""
    segs = [seg("S01", 0, 10, "Welcome everyone, and thank you for joining. Your participation "
                "is completely voluntary, and you may skip any question."),
            seg("S02", 10, 16, "Thanks. Let's start. What about you, [first_name_1]? Where do you run into trouble?"),
            seg("S03", 16, 37, "Well, usually I make coffee first, and then I try to find my keys."),
            seg("S01", 37, 44, "That makes sense. And you, [first_name_2]? How do you get around?"),
            seg("S04", 44, 65, "I mostly use my cane and an app on my phone that reads labels out loud."),
            seg("S02", 65, 71, "Okay, let's move on to the next question. What would you want?")]
    r = assign(segs)
    assert r["mode"] == "group" and not r["uncertain"]
    assert r["roles"]["S01"].startswith("Moderator") and r["roles"]["S02"].startswith("Moderator")


def test_known_voice_does_not_demote_the_other_moderator(tmp_path):
    lib = tmp_path / "voices.json"
    voice_library.remember("Moderator A", "Moderator 1", np.array([1.0, 0, 0, 0]), path=lib)
    segs = [seg("S01", 0, 10, "Welcome everyone, and thank you for joining. Your participation "
                "is completely voluntary, and you may skip any question."),
            seg("S02", 10, 16, "Thanks. Let's start. What about you, [first_name_1]? Where do you run into trouble?"),
            seg("S03", 16, 37, "Well, usually I make coffee first, and then I try to find my keys."),
            seg("S01", 37, 44, "That makes sense. And you, [first_name_2]? How do you get around?"),
            seg("S04", 44, 65, "I mostly use my cane and an app on my phone that reads labels out loud."),
            seg("S02", 65, 71, "Okay, let's move on to the next question. What would you want?")]
    voices = {"S01": np.array([1.0, 0, 0, 0]), "S02": np.array([0, 1.0, 0, 0]),
              "S03": np.array([0, 0, 1.0, 0]), "S04": np.array([0, 0, 0, 1.0])}
    r = assign(segs, voices=voices, library=voice_library.load(lib))
    assert r["roles"]["S01"].startswith("Moderator") and r["roles"]["S02"].startswith("Moderator")
    assert r["roles"]["S03"].startswith("Participant") and r["roles"]["S04"].startswith("Participant")
