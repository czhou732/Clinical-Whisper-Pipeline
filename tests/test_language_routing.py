"""Masking, scoring and keyword screening follow the recording's language."""

import json

import pytest

import addons
import inference_pipeline as ip

EN = {"code": "en", "name": "English", "confidence": 0.9, "other_share": 0.0}
ES = {"code": "es", "name": "Spanish", "confidence": 0.9, "other_share": 0.0}
RU = {"code": "ru", "name": "Russian", "confidence": 1.0, "other_share": 0.0}


def test_english_uses_the_default_masker():
    assert ip._masker_for(EN) == {}


def test_other_language_without_the_addon_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(addons, "LANGUAGES_ROOT", tmp_path / "none")
    with pytest.raises(RuntimeError, match="Spanish language pack"):
        ip._masker_for(ES)


def test_language_the_masker_never_learned_is_refused(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="can't mask names in Russian"):
        ip._masker_for(RU)


def test_other_language_with_the_addon_uses_the_multilingual_masker(monkeypatch, tmp_path):
    folder = tmp_path / "pack" / "openmed" / addons.LANGUAGES_MODEL.replace("/", "_")
    folder.mkdir(parents=True)
    for f in ("weights.safetensors", "config.json"):
        (folder / f).write_text("x")
    root = tmp_path / "support"
    addons.install_languages(tmp_path / "pack", root=root)
    monkeypatch.setattr(addons, "LANGUAGES_ROOT", root)
    kwargs = ip._masker_for(ES)
    assert kwargs["model_name"] == addons.LANGUAGES_MODEL
    assert kwargs["lang"] == "es"
    assert kwargs["cache_dir"] == str(root / "openmed")


def test_install_any_tells_the_two_addons_apart(tmp_path, monkeypatch):
    folder = tmp_path / "pack" / "openmed" / addons.LANGUAGES_MODEL.replace("/", "_")
    folder.mkdir(parents=True)
    for f in ("weights.safetensors", "config.json"):
        (folder / f).write_text("x")
    monkeypatch.setattr(addons, "LANGUAGES_ROOT", tmp_path / "support")
    assert addons.install_any(tmp_path / "pack") == "languages"
    with pytest.raises(ValueError):
        addons.install_any(tmp_path / "empty-dir-that-is-not-there")


def test_non_english_skips_scores_keywords_and_fillers(monkeypatch, tmp_path):
    import llm_clinical_scorer

    def _must_not_run(*a, **k):
        raise AssertionError("scored a Spanish recording")

    monkeypatch.setattr(llm_clinical_scorer, "score_transcript", _must_not_run)
    monkeypatch.setattr(ip, "_scoring_installed", lambda cfg: True)
    cfg = {"llm_scoring": {"enabled": True},
           "pipeline": {"analysis_output_folder": str(tmp_path)}, "audio_retention": "keep"}
    state = {
        "job": {"job_id": "t3"}, "job_id": "t3", "file_path": tmp_path / "a.wav",
        "original_filename": "a.wav",
        "segments": [{"speaker": "S01", "start": 0.0, "end": 4.0, "text": "¿Cómo se siente?"},
                     {"speaker": "S02", "start": 4.0, "end": 9.0, "text": "Eh, pues me quiero morir."}],
        "transcript": "...",
        "stats": {"word_count": 7, "duration_seconds": 9.0, "language": ES},
        "overall_acoustics": {}, "speaker_acoustics": {}, "audio_stats": None, "warnings": [],
    }
    payload = json.loads(open(ip.InferencePipeline(cfg).score_job(state)).read())
    assert payload["language"]["code"] == "es"
    assert payload["llm_clinical_scoring"] == {}
    assert payload["status"] == "completed"          # a choice, not a fault
    assert "English-only" in payload["clinical_review"]["note"]
    assert payload["clinical_review"]["items"] == []
    assert all(p["filler_rate"] is None for p in payload["timing_features"]["per_speaker"].values())
    assert any("built and checked on English" in n for n in payload["notes"])


def test_language_pack_wins_and_uses_a_lower_threshold(monkeypatch, tmp_path):
    model = addons.LANGUAGE_PACKS["es"]
    pack = tmp_path / "ClinicalWhisper Spanish"
    conv = pack / "openmed" / model.replace("/", "_")
    conv.mkdir(parents=True)
    (conv / "weights.safetensors").write_text("x")
    (pack / "hub" / ("models--" + model.replace("/", "--"))).mkdir(parents=True)
    monkeypatch.setattr(addons, "pack_root", lambda code: tmp_path / "support" / f"lang-{code}")
    assert addons.install_any(pack) == "lang-es"
    kwargs = ip._masker_for(ES)
    assert kwargs["model_name"] == model and kwargs["lang"] == "es"
    assert kwargs["confidence_threshold"] == 0.5
