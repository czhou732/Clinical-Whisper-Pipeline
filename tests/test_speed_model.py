"""The time estimate: defaults before any run, then learned from this Mac."""

import pytest

import speed_model


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(speed_model, "_path", lambda: tmp_path / "speed.json")
    return tmp_path / "speed.json"


def test_defaults_are_slower_on_small_macs(store, monkeypatch):
    monkeypatch.setattr(speed_model, "_ram_gb", lambda: 64.0)
    fast = speed_model.load()
    monkeypatch.setattr(speed_model, "_ram_gb", lambda: 16.0)
    slow = speed_model.load()
    assert slow["transcribe"] == pytest.approx(fast["transcribe"] * speed_model._SLOW_FACTOR)
    assert fast["runs"] == 0


def test_first_run_replaces_default_then_averages(store):
    speed_model.record("transcribe", 3600, 360)  # 0.1
    assert speed_model.load()["transcribe"] == pytest.approx(0.1)
    speed_model.record("transcribe", 3600, 720)  # 0.2
    assert speed_model.load()["transcribe"] == pytest.approx(0.7 * 0.1 + 0.3 * 0.2)
    assert speed_model.load()["runs"] == 2


def test_short_files_are_not_learned_from(store):
    speed_model.record("transcribe", 30, 60)
    assert not store.exists()


def test_estimate_includes_scoring_only_when_asked_and_is_rough_at_first(store, monkeypatch):
    monkeypatch.setattr(speed_model, "_ram_gb", lambda: 64.0)
    t_only, rough = speed_model.estimate(3600, scoring=False)
    both, _ = speed_model.estimate(3600, scoring=True)
    assert rough
    assert t_only == pytest.approx(3600 * speed_model._FAST["transcribe"])
    assert both == pytest.approx(3600 * sum(speed_model._FAST.values()))


def test_first_scoring_measurement_replaces_default_even_after_transcription(store):
    speed_model.record("transcribe", 3600, 360)
    speed_model.record("score", 3600, 1800)  # 0.5, far slower than the default
    assert speed_model.load()["score"] == pytest.approx(0.5)
