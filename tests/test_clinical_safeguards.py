"""Intended-use notice, score reliability, recording-quality flags."""

from clinical_safeguards import assess_quality, level_stats, score_reliability, spearman_brown


def test_spearman_brown_matches_the_reliability_report():
    # evals/reports/reliability.md: hesitancy ICC(1,1) 0.288 -> mean of 5 = 0.67
    assert round(spearman_brown(0.288, 5), 2) == 0.67
    rel = score_reliability(5)
    assert rel["elaboration_positive"]["icc"] == 0.88 and rel["elaboration_positive"]["adequate"]
    assert not rel["psychomotor_indicators"]["adequate"]
    assert "subject_speech_rate_wps" in rel["psychomotor_indicators"]["use_instead"]
    # Latency is not reliable in this version, so it is never suggested.
    assert "subject_response_latency_median_s" not in rel["psychomotor_indicators"]["use_instead"]


def _seg(spk, start, end, **kw):
    return {"speaker": spk, "start": start, "end": end, "text": "x", **kw}


def test_clean_interview_has_no_flags():
    segs = [_seg("S01", 0, 200), _seg("S02", 200, 600)]
    q = assess_quality(segs, "S02", level_stats(0.01 * 1000, 1000, 0, 16000) | {"duration_s": 600},
                       expected_speakers=2)
    assert q["flags"] == []
    assert q["participant_speech_s"] == 400


def test_problem_recording_raises_each_flag():
    segs = [_seg("S01", 0, 30), _seg("S02", 30, 50), _seg("S03", 50, 55, speaker_uncertain=True)]
    stats = {"duration_s": 55, "rms_dbfs": -48.0, "clipped_fraction": 0.01}
    codes = {f["code"] for f in assess_quality(segs, "S02", stats, expected_speakers=2)["flags"]}
    assert codes == {"short_recording", "little_participant_speech", "clipping",
                     "quiet_recording", "brief_speakers", "speaker_count"}


def test_level_stats_from_running_totals():
    s = level_stats(sum_squares=0.01 * 16000, count=16000, clipped=16, sample_rate=16000)
    assert s["duration_s"] == 1.0
    assert round(s["rms_dbfs"]) == -20
    assert s["clipped_fraction"] == 0.001


def test_parallel_speaker_features_match_sequential(tmp_path, monkeypatch):
    """Extracting speakers side by side must not change a single number."""
    import numpy as np
    import pytest

    sf = pytest.importorskip("soundfile")
    acoustic_features = pytest.importorskip("acoustic_features")
    import inference_pipeline as ip

    extractor = acoustic_features.AcousticExtractor()
    if not extractor.is_available():
        pytest.skip("OpenSMILE not installed")
    sr = 16000
    t = np.arange(sr * 12) / sr
    # Two "voices": different pitch and loudness, alternating every 3 s.
    audio = np.where((t // 3) % 2 == 0,
                     0.3 * np.sin(2 * np.pi * 140 * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t)),
                     0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t)))
    path = tmp_path / "two_voices.wav"
    sf.write(path, audio.astype(np.float32), sr, subtype="PCM_16")
    segs = [{"speaker": "A" if i % 2 == 0 else "B", "start": 3.0 * i, "end": 3.0 * (i + 1)}
            for i in range(4)]

    pipe = ip.InferencePipeline({})
    monkeypatch.setattr(ip, "ACOUSTIC_WORKERS", 1)
    sequential = pipe._extract_speaker_acoustics(extractor, path, segs)
    monkeypatch.setattr(ip, "ACOUSTIC_WORKERS", 2)
    parallel = pipe._extract_speaker_acoustics(extractor, path, segs)
    assert sequential == parallel
    assert set(parallel) == {"A", "B"}
