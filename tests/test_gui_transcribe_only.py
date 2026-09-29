"""The app's transcribe-only switch: the flag must reach the batch config."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
soundfile = pytest.importorskip("soundfile")
from fastapi.testclient import TestClient  # noqa: E402

import gui_server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(gui_server, "INPUT_DIR", tmp_path)
    started: list = []
    # Don't run the pipeline: this checks what the batch was configured with.
    monkeypatch.setattr(gui_server.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())
    # The server only answers to this machine's hostnames.
    return TestClient(gui_server.app, base_url="http://127.0.0.1"), started


def _wav(tmp_path):
    path = tmp_path / "clip.wav"
    soundfile.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return path


def test_flag_is_recorded_when_checked(client, tmp_path):
    api, _ = client
    with open(_wav(tmp_path), "rb") as fh:
        r = api.post("/api/upload", files={"files": ("clip.wav", fh, "audio/wav")},
                     data={"transcribe_only": "1"})
    batch = gui_server._batches[r.json()["batch_id"]]
    assert batch["transcribe_only"] is True


def test_scoring_stays_on_by_default(client, tmp_path):
    api, _ = client
    with open(_wav(tmp_path), "rb") as fh:
        r = api.post("/api/upload", files={"files": ("clip.wav", fh, "audio/wav")})
    batch = gui_server._batches[r.json()["batch_id"]]
    assert batch["transcribe_only"] is False


def test_deliberate_skip_is_a_note_not_a_warning(monkeypatch, tmp_path):
    """A run the user asked to skip scoring for is 'completed', not 'with warnings'."""
    import inference_pipeline as ip

    cfg = {"llm_scoring": {"enabled": False, "skipped_by_request": True},
           "pipeline": {"analysis_output_folder": str(tmp_path)},
           "audio_retention": "keep"}
    pipeline = ip.InferencePipeline(cfg)
    state = {
        "job": {"job_id": "t1"}, "job_id": "t1", "file_path": tmp_path / "a.wav",
        "original_filename": "a.wav",
        "segments": [{"speaker": "S01", "start": 0.0, "end": 4.0, "text": "hello there"},
                     {"speaker": "S02", "start": 4.0, "end": 9.0, "text": "hello back"}],
        "transcript": "hello there hello back",
        "stats": {"word_count": 4, "duration_seconds": 9.0},
        "overall_acoustics": {}, "speaker_acoustics": {}, "audio_stats": None,
        "warnings": [],
    }
    import json
    payload = json.loads(open(pipeline.score_job(state)).read())
    assert payload["status"] == "completed"
    assert payload["warnings"] == []
    assert any("skipped" in n for n in payload["notes"])



def test_request_from_another_site_is_refused(client, tmp_path):
    api, _ = client
    with open(_wav(tmp_path), "rb") as fh:
        r = api.post("/api/upload", files={"files": ("clip.wav", fh, "audio/wav")},
                     headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_request_from_the_apps_own_page_is_accepted(client, tmp_path):
    api, _ = client
    with open(_wav(tmp_path), "rb") as fh:
        r = api.post("/api/upload", files={"files": ("clip.wav", fh, "audio/wav")},
                     headers={"Origin": "http://127.0.0.1"})
    assert r.status_code == 200


def test_foreign_hostname_is_refused():
    """DNS rebinding: a site pointing its own name at 127.0.0.1 is still refused."""
    api = TestClient(gui_server.app, base_url="http://evil.example")
    assert api.get("/api/diagnostics").status_code == 400


def test_in_place_batch_reads_originals_and_never_moves_them(client, tmp_path):
    _, started = client
    wav = _wav(tmp_path)
    result = gui_server.start_batch_from_paths([str(wav)], transcribe_only=True)
    batch = gui_server._batches[result["batch_id"]]
    assert batch["in_place"] is True and batch["transcribe_only"] is True
    assert started[-1]["args"][1] == [wav.resolve()]   # the original path, not a copy
    assert wav.exists()


def test_in_place_batch_rejects_missing_and_non_audio_files(client, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio")
    with pytest.raises(ValueError, match="Not a supported audio file"):
        gui_server.start_batch_from_paths([str(notes)])
    with pytest.raises(ValueError, match="Not a file"):
        gui_server.start_batch_from_paths([str(tmp_path / "missing.wav")])


def test_in_place_batch_runs_with_audio_retention_keep(tmp_path, monkeypatch):
    """The pipeline normally archives (moves) its input; for in-place files it must not."""
    import inference_pipeline

    seen = {}

    class FakePipeline:
        def __init__(self, cfg, **kw):
            seen["cfg"] = cfg

        def transcribe_jobs(self, jobs):
            return [RuntimeError("stop here") for _ in jobs]

        def release_transcriber(self):
            pass

        def release_all(self):
            pass

    monkeypatch.setattr(inference_pipeline, "InferencePipeline", FakePipeline)
    wav = _wav(tmp_path)
    batch_id = gui_server._new_batch([wav])
    gui_server._batches[batch_id]["in_place"] = True
    gui_server.process_batch_task(batch_id, [wav])
    assert seen["cfg"]["audio_retention"] == "keep"
    assert wav.exists()


def test_uploaded_files_with_the_same_name_do_not_overwrite(client, tmp_path):
    api, started = client
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    soundfile.write(a / "session.wav", np.full(16000, 0.1, dtype=np.float32), 16000)
    soundfile.write(b / "session.wav", np.full(16000, 0.3, dtype=np.float32), 16000)
    with open(a / "session.wav", "rb") as fa, open(b / "session.wav", "rb") as fb:
        r = api.post("/api/upload", files=[("files", ("session.wav", fa, "audio/wav")),
                                           ("files", ("session.wav", fb, "audio/wav"))])
    assert r.status_code == 200
    saved = started[-1]["args"][1]
    assert len(saved) == 2 and saved[0] != saved[1]
    assert soundfile.read(saved[0])[0].mean() != soundfile.read(saved[1])[0].mean()
