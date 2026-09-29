"""The 'stopped while processing' notice, and per-job temporary audio."""

import json

import numpy as np
import pytest


@pytest.fixture
def diag(tmp_path, monkeypatch):
    import crash_diagnostics as cd

    monkeypatch.setattr(cd, "SESSION_PATH", tmp_path / "session.json")
    monkeypatch.setattr(cd, "_previous_crash", None)
    return cd


def _previous(cd, **fields):
    cd.SESSION_PATH.write_text(json.dumps({"pid": 999999999, **fields}))  # a dead pid
    cd.start_session()
    return cd.previous_crash()


def test_interrupted_batch_is_reported(diag):
    assert _previous(diag, working=True, stage="Transcribing")["stage"] == "Transcribing"


def test_idle_app_that_was_quit_is_not_reported(diag):
    # A stage name left over after the work ended is not evidence of a crash.
    assert _previous(diag, working=False, stage="Acoustics") is None


def test_marker_after_a_batch_says_idle(diag):
    diag.start_session()
    diag.begin_work("3 file(s)")
    assert json.loads(diag.SESSION_PATH.read_text())["working"] is True
    diag.end_work()
    assert json.loads(diag.SESSION_PATH.read_text())["working"] is False


def test_same_named_recordings_get_separate_temporary_audio(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    import inference_pipeline as ip

    monkeypatch.setattr(ip, "_SCRATCH_ROOT", tmp_path / "scratch")
    a, b = tmp_path / "P01", tmp_path / "P02"
    a.mkdir(), b.mkdir()
    sf.write(a / "session.wav", np.full(16000, 0.1, dtype=np.float32), 16000)
    sf.write(b / "session.wav", np.full(16000, 0.3, dtype=np.float32), 16000)

    pipe = ip.InferencePipeline({})
    asr_a, _, _ = pipe._preprocess_audio(a / "session.wav", tag="job1")
    asr_b, _, _ = pipe._preprocess_audio(b / "session.wav", tag="job2")
    assert asr_a != asr_b
    assert asr_a.exists() and asr_b.exists()   # neither overwrote the other
