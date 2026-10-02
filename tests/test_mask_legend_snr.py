"""The masking key and the SNR estimate."""

import numpy as np
import pytest

from mask_legend import legend


def test_legend_explains_tags_without_revealing_anything():
    text = "[first_name_1] met [first_name_2] in [city_1] on [date_1] and [date_2]."
    key = legend(text)
    assert "2 different first names: [first_name_1], [first_name_2]" in key
    assert "1 city: [city_1]" in key
    assert "same tag means the same word" in key


def test_legend_is_empty_without_tags():
    assert legend("Nothing was hidden here.") == ""


def test_snr_tracks_added_noise(tmp_path):
    sf = pytest.importorskip("soundfile")
    from snr import estimate_snr

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr * 120) / sr
    voice = np.zeros_like(t, dtype=np.float32)
    segments = []
    for start in range(0, 120, 6):  # 4 s of "speech", 2 s of silence
        idx = (t >= start) & (t < start + 4)
        voice[idx] = 0.1 * np.sin(2 * np.pi * 180 * t[idx])
        segments.append({"start": float(start), "end": float(start + 4)})
    speech_rms = np.sqrt(np.mean(voice[voice != 0] ** 2))
    for target in (30, 10):
        noise = rng.standard_normal(len(t)) * speech_rms / 10 ** (target / 20)
        path = tmp_path / f"n{target}.wav"
        sf.write(path, (voice + noise).astype(np.float32), sr, subtype="FLOAT")
        assert estimate_snr(str(path), segments) == pytest.approx(target, abs=2.5)


def test_noisy_recording_is_flagged():
    from clinical_safeguards import assess_quality

    segs = [{"speaker": "S01", "start": 0, "end": 400}, {"speaker": "S02", "start": 400, "end": 800}]
    noisy = assess_quality(segs, "S02", {"duration_s": 800, "snr_db": 9.0})
    clean = assess_quality(segs, "S02", {"duration_s": 800, "snr_db": 30.0})
    assert "noisy_recording" in {f["code"] for f in noisy["flags"]}
    assert "noisy_recording" not in {f["code"] for f in clean["flags"]}
    assert noisy["snr_db"] == 9.0
