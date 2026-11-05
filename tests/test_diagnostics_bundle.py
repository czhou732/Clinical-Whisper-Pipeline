"""The diagnostics zip must not carry file names, paths or the account name."""

import json
import zipfile

import crash_diagnostics
import diagnostics_bundle
from diagnostics_bundle import redact


def test_redact_removes_paths_file_names_and_user():
    line = ("Job gui_1: wrote /Users/ellen/Documents/journal club audio/P07 smith.wav done\n"
            "Error opening '/var/folders/d9/x/P07_16k.wav': no such file\n"
            "Stage: Transcribing P07 smith intake.m4a\n"
            "home is ellen's")
    out = redact(line, user="ellen")
    for leaked in ("ellen", "smith", "P07", "journal club", "/Users", "/var"):
        assert leaked not in out
    assert "Job gui_1: wrote <path>" in out
    assert "Stage: <file>" in out


def test_redact_keeps_what_debugging_needs():
    line = "Round 3/6: 4 windows, 12.4 s\nMemoryError: cannot allocate 4.9 GB"
    assert redact(line, user="nobody") == line


def test_save_writes_readme_machine_and_redacted_logs(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "clinicalwhisper.log").write_text("wrote /Users/x/Output/P01.json\n")
    (logs / "clinicalwhisper.log.1").write_text("Decoding P02 interview.wav\n")
    (logs / "unrelated.png").write_bytes(b"\x89PNG")
    (logs / "other_tool.log").write_text("someone else's log\n")
    monkeypatch.setattr(crash_diagnostics, "LOG_DIR", logs)
    monkeypatch.setattr(crash_diagnostics, "LOG_PATH", logs / "clinicalwhisper.log")
    out = diagnostics_bundle.save(tmp_path / "out", {"version": "5.2.0"})
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert names == {"README.txt", "machine.json", "logs/clinicalwhisper.log",
                         "logs/clinicalwhisper.log.1"}
        body = zf.read("logs/clinicalwhisper.log").decode() + zf.read("logs/clinicalwhisper.log.1").decode()
        assert "P01" not in body and "P02" not in body
        assert json.loads(zf.read("machine.json"))["version"] == "5.2.0"
