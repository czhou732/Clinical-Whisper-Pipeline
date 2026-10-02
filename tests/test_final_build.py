"""Progress, sync detection, keep-awake and the small-Mac scoring default."""

import sys
from pathlib import Path

import pandas as pd
import pytest


def test_progress_reports_stage_and_batch_time_left():
    from progress_report import BatchProgress

    now = [0.0]
    p = BatchProgress(total_files=4, total_audio_s=4 * 3600, interval=30, clock=lambda: now[0])
    assert "Transcribing" in p.update("Transcribing", 0.0)
    now[0] = 600
    line = p.update("Transcribing", 0.5)          # 10 min for half: 10 min left
    assert "50%" in line and "about 10 min left in this stage" in line
    p.file_done(3600)                              # 1 h of audio took 10 min
    now[0] = 700
    line = p.update("De-identifying")
    assert "1/4 files done" in line and "left in the batch" in line
    assert p.update("De-identifying") == ""        # throttled within the interval


def test_known_sync_folders_are_detected(tmp_path):
    from sync_check import synced_by

    home = tmp_path
    for name, expected in [("Library/CloudStorage/OneDrive-Yale", "cloud storage"),
                           ("Dropbox/study", "Dropbox"),
                           ("OneDrive - Yale University/x", "OneDrive")]:
        folder = home / name
        folder.mkdir(parents=True)
        assert expected in synced_by(folder, home=home)
    local = home / "ClinicalWhisper"
    local.mkdir()
    assert synced_by(local, home=home) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS caffeinate")
def test_keep_awake_starts_and_stops_caffeinate():
    import subprocess
    import os
    from keep_awake import keep_awake

    def ours():
        out = subprocess.run(["pgrep", "-f", f"caffeinate -i -s -w {os.getpid()}"],
                             capture_output=True, text=True).stdout
        return bool(out.strip())

    with keep_awake():
        assert ours()
    import time
    time.sleep(0.3)
    assert not ours()


def test_small_mac_defaults_to_transcribe_only(tmp_path, monkeypatch):
    import batch_processor

    seen = {}
    monkeypatch.setattr(batch_processor, "_ram_gib", lambda: 8.0)
    monkeypatch.setattr(batch_processor.offline, "network_allowed", lambda: True)
    monkeypatch.setattr(batch_processor, "batch_process",
                        lambda *a, **k: seen.update(k) or pd.DataFrame([{"x": 1}]))
    batch_processor.main(["-i", str(tmp_path), "-o", str(tmp_path / "s.csv")])
    assert seen["transcribe_only"] is True
    batch_processor.main(["-i", str(tmp_path), "-o", str(tmp_path / "s.csv"), "--score"])
    assert seen["transcribe_only"] is False


def test_short_extra_label_cannot_become_the_interviewer():
    """Host split into a main voice plus a fragment: the fragment must not win."""
    from transcript_formatter import classify_speakers

    segs = []
    t = 0.0
    for i in range(40):  # host (S02) asks, guest (S01) answers at length
        segs.append({"speaker": "S02", "start": t, "end": t + 4, "text": "What do you think about that?"})
        segs.append({"speaker": "S01", "start": t + 4, "end": t + 40,
                     "text": " ".join(["the evidence suggests"] * 25) + "."})
        t += 40
    segs.append({"speaker": "S03", "start": t, "end": t + 2, "text": "Right?"})   # fragment
    roles = classify_speakers(segs)
    assert roles["S02"] == "Interviewer"
    assert roles["S01"] == "Subject"
    assert roles["S03"] == "Other_1"


def test_resume_scores_a_saved_transcript_without_transcribing_again(tmp_path, monkeypatch):
    import json
    import batch_processor
    from inference_pipeline import SCORING_PENDING

    audio_dir, analysis = tmp_path / "in", tmp_path / "out"
    audio_dir.mkdir(), analysis.mkdir()
    rec = audio_dir / "P01.wav"
    rec.write_bytes(b"not decoded in this test")
    (analysis / "job1_analysis.json").write_text(json.dumps({
        "job_id": "job1", "status": SCORING_PENDING, "participant_id": "P01",
        "source_audio": {"stored_path": str(rec), "original_filename": "P01.wav"},
        "segments": [], "transcript": "hello", "statistics": {"word_count": 1},
        "quality": {"duration_s": 5.0}, "warnings": [],
    }))
    out = tmp_path / "summary.csv"
    out.write_text("filename,word_count\nother.wav,3\n")   # an earlier run's rows

    calls = {"scored": [], "transcribed": []}

    class FakePipeline:
        def __init__(self, cfg, **kw): pass
        def score_job(self, state):
            calls["scored"].append(state["job_id"])
            final = analysis / "job1_analysis.json"
            final.write_text(json.dumps({"job_id": "job1", "status": "completed",
                                         "statistics": {"word_count": 1}}))
            return str(final)
        def iter_transcribed(self, jobs):
            calls["transcribed"].extend(jobs)
            return iter(())
        def release_scorer(self): pass
        def release_transcriber(self): pass
        def release_all(self): pass

    monkeypatch.setattr(batch_processor, "InferencePipeline", FakePipeline)
    monkeypatch.setattr(batch_processor, "resolve_path", lambda p: str(analysis))
    df = batch_processor.batch_process(str(audio_dir), str(out), resume=True)
    assert calls["scored"] == ["job1"]
    assert calls["transcribed"] == []                 # nothing re-transcribed
    assert list(df["filename"]) == ["P01.wav"]
