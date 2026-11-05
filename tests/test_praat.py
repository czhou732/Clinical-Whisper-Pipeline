"""Praat voice measures (add-on) and the add-on's install path."""

import sys

import numpy as np
import pytest

import addons
import praat_measures as pm


def test_install_finds_the_site_folder(tmp_path, monkeypatch):
    src = tmp_path / "ClinicalWhisper Praat"
    (src / "site").mkdir(parents=True)
    (src / "site" / "parselmouth.cpython-313-darwin.so").write_bytes(b"x")
    (src / "LICENSE-GPL3.txt").write_text("GPL")
    root = tmp_path / "support" / "praat"
    addons.install_praat(src, root=root)
    assert (root / "site" / "parselmouth.cpython-313-darwin.so").exists()
    assert (root / "LICENSE-GPL3.txt").exists()
    monkeypatch.setattr(sys, "path", list(sys.path))
    addons.add_praat_path(root)
    assert str(root / "site") in sys.path


def test_speaker_audio_samples_evenly(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000 * 100, dtype=np.float32), 16000)
    segs = [{"speaker": "A", "start": float(i), "end": i + 0.9} for i in range(0, 100, 2)]
    audio = pm.speaker_audio(str(wav), segs, "A", max_s=10)
    assert 8 * 16000 <= len(audio) <= 12 * 16000


@pytest.mark.skipif(not pm.available(), reason="parselmouth not installed here")
def test_measures_on_a_synthetic_voice():
    t = np.arange(16000 * 3) / 16000
    tone = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)
    out = pm.measure(tone.astype(np.float32))
    assert out["pitch_floor_hz"] == 60.0
    assert 110 < out["f0_mean_hz"] < 130
    assert out["hnr_db_mean"] > 20  # a clean periodic signal
