"""Naming speakers by hand and correcting their roles in the window."""

import json

import pytest

from transcript_formatter import (clean_names, format_structured_transcript,
                                  speaker_samples)

SEGMENTS = [
    {"speaker": "S01", "start": 0.0, "end": 5.0, "text": "Thanks for joining, let's start with your morning"},
    {"speaker": "S02", "start": 5.0, "end": 30.0, "text": "I usually get up and turn on the coffee maker first"},
    {"speaker": "S03", "start": 30.0, "end": 31.0, "text": "Yeah."},
    {"speaker": "S02", "start": 31.0, "end": 40.0, "text": "Then I pack my bag and check I have my tools"},
]


def test_named_speaker_keeps_role_on_every_line():
    text = format_structured_transcript(SEGMENTS, {"S01": "Interviewer", "S02": "Subject"},
                                        {"S02": "P01"})
    assert "] Interviewer: Thanks" in text
    assert "] P01 (Subject): I usually" in text


def test_labels_are_cleaned():
    names = clean_names({"S01": "  Moderator:  A ", "S02": "x" * 80, "S03": "",
                         "S99": "ghost", "S04": 5}, {"S01", "S02", "S03", "S04"})
    assert names == {"S01": "Moderator A", "S02": "x" * 40}


def test_samples_skip_short_replies_and_sum_talk_time():
    s = speaker_samples(SEGMENTS)
    assert s["S02"]["talk_s"] == 34.0
    assert len(s["S02"]["lines"]) == 2
    assert s["S03"]["lines"] == []


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import gui_server  # noqa: E402


@pytest.fixture
def batch(tmp_path, monkeypatch):
    path = tmp_path / "a_analysis.json"
    analysis = {"speaker_roles": {"S01": "Interviewer", "S02": "Subject", "S03": "Other_1"},
                "segments": SEGMENTS, "llm_clinical_scoring": {"hesitancy_score": 3},
                "quality": {}, "timing_features": {}}
    path.write_text(json.dumps(analysis))
    gui_server._batches["b1"] = {
        "transcribe_only": False, "log": [],
        "files": [{"filename": "a.wav", "segments": SEGMENTS, "analysis": analysis,
                   "analysis_path": str(path), "result": {}}],
    }
    calls = []

    def fake_score(*args, **kwargs):
        calls.append(args)
        return {"hesitancy_score": 4}

    import llm_clinical_scorer
    monkeypatch.setattr(llm_clinical_scorer, "score_transcript", fake_score)
    yield TestClient(gui_server.app, base_url="http://127.0.0.1"), path, calls
    gui_server._batches.pop("b1", None)


def test_names_only_relabels_without_rescoring(batch):
    api, path, calls = batch
    r = api.post("/api/rescore/b1/0", json={
        "roles": {"S01": "Interviewer", "S02": "Subject", "S03": "Other"},
        "names": {"S01": "Moderator A", "S02": "P01"}})
    data = r.json()
    assert data["status"] == "success" and data["rescored"] is False
    assert calls == []
    saved = json.loads(path.read_text())
    assert saved["speaker_names"] == {"S01": "Moderator A", "S02": "P01"}
    assert "P01 (Subject):" in saved["structured_transcript"]
    assert saved["llm_clinical_scoring"] == {"hesitancy_score": 3}


def test_role_change_rescores(batch):
    api, _, calls = batch
    r = api.post("/api/rescore/b1/0", json={
        "roles": {"S01": "Subject", "S02": "Interviewer", "S03": "Other"}, "names": {}})
    assert r.json()["rescored"] is True
    assert len(calls) == 1


def test_two_participants_are_refused(batch):
    api, _, calls = batch
    r = api.post("/api/rescore/b1/0", json={
        "roles": {"S01": "Subject", "S02": "Subject", "S03": "Other"}, "names": {}})
    assert r.status_code == 400
    assert "one speaker" in r.json()["message"]
    assert calls == []
