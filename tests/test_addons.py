"""The clinical scoring add-on: finding, installing and resolving the model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import addons  # noqa: E402

MODEL = addons.SCORING_MODEL
NAME = "models--" + MODEL.replace("/", "--")


def _fake_hub(hub: Path, rev: str = "abc123") -> Path:
    base = hub / NAME
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text(rev)
    snap = base / "snapshots" / rev
    snap.mkdir(parents=True)
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        (snap / f).write_text("x")
    return snap


def _addon_volume(tmp_path: Path) -> Path:
    volume = tmp_path / "Volumes" / "ClinicalWhisper Scoring"
    _fake_hub(volume / "ClinicalWhisper Scoring" / "hub")
    return volume


def test_install_from_mounted_volume(tmp_path):
    root = tmp_path / "support"
    path = addons.install(_addon_volume(tmp_path), root=root)
    assert path == root / "hub" / NAME / "snapshots" / "abc123"
    assert addons.installed_path(root=root) == path


@pytest.mark.parametrize("pick", ["volume", "folder", "hub"])
def test_any_reasonable_folder_choice_works(tmp_path, pick):
    volume = _addon_volume(tmp_path)
    chosen = {"volume": volume,
              "folder": volume / "ClinicalWhisper Scoring",
              "hub": volume / "ClinicalWhisper Scoring" / "hub"}[pick]
    assert addons.find_in(chosen) == volume / "ClinicalWhisper Scoring" / "hub"


def test_wrong_folder_is_refused_and_installs_nothing(tmp_path):
    root = tmp_path / "support"
    (tmp_path / "Downloads").mkdir()
    with pytest.raises(ValueError):
        addons.install(tmp_path / "Downloads", root=root)
    assert addons.installed_path(root=root) is None


def test_incomplete_model_is_not_usable(tmp_path):
    snap = _fake_hub(tmp_path / "hub")
    (snap / "model.safetensors").unlink()
    assert addons.find_in(tmp_path) is None


def test_reinstall_replaces_the_old_copy(tmp_path):
    root = tmp_path / "support"
    addons.install(_addon_volume(tmp_path), root=root)
    newer = tmp_path / "newer"
    _fake_hub(newer / "hub", rev="def456")
    path = addons.install(newer, root=root)
    assert path.name == "def456"
    assert not (root / "hub" / NAME / "snapshots" / "abc123").exists()
    assert not [p for p in root.iterdir() if p.name.startswith(".")]


def test_resolve_prefers_installed_addon(tmp_path, monkeypatch):
    root = tmp_path / "support"
    monkeypatch.setattr(addons, "ADDON_ROOT", root)
    monkeypatch.setattr(addons, "_bundle_hub", lambda: None)
    assert addons.resolve(MODEL) == MODEL  # nothing installed: the plain name
    path = addons.install(_addon_volume(tmp_path), root=root)
    assert addons.resolve(MODEL) == str(path)


def test_packaged_app_without_addon_has_no_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(addons, "ADDON_ROOT", tmp_path / "support")
    monkeypatch.setattr(addons, "_bundle_hub", lambda: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert addons.scoring_available() is False
    addons.install(_addon_volume(tmp_path), root=tmp_path / "support")
    assert addons.scoring_available() is True


def test_source_checkout_always_has_scoring(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert addons.scoring_available() is True


def test_missing_addon_blanks_scores_instead_of_faking_them(monkeypatch, tmp_path):
    """Without the add-on, scoring is not attempted and the result says why."""
    import json

    import inference_pipeline as ip
    import llm_clinical_scorer

    monkeypatch.setattr(ip, "_scoring_installed", lambda cfg: False)

    def _must_not_run(*a, **k):
        raise AssertionError("scoring ran without the model")

    monkeypatch.setattr(llm_clinical_scorer, "score_transcript", _must_not_run)
    cfg = {"llm_scoring": {"enabled": True},
           "pipeline": {"analysis_output_folder": str(tmp_path)},
           "audio_retention": "keep"}
    state = {
        "job": {"job_id": "t2"}, "job_id": "t2", "file_path": tmp_path / "a.wav",
        "original_filename": "a.wav",
        "segments": [{"speaker": "S01", "start": 0.0, "end": 4.0, "text": "how are you"},
                     {"speaker": "S02", "start": 4.0, "end": 9.0, "text": "fine thanks"}],
        "transcript": "how are you fine thanks",
        "stats": {"word_count": 5, "duration_seconds": 9.0},
        "overall_acoustics": {}, "speaker_acoustics": {}, "audio_stats": None,
        "warnings": [],
    }
    payload = json.loads(open(ip.InferencePipeline(cfg).score_job(state)).read())
    assert payload["llm_clinical_scoring"] == {}
    assert payload["status"] == "completed_with_warnings"
    assert len(payload["warnings"]) == 1 and "isn't installed" in payload["warnings"][0]
