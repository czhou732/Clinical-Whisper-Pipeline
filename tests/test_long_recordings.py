"""Long clinical recordings: speaker-count limit, streamed audio, numbered
de-identification tags, and the batch command's ID mapping and resume."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from moss_chunking import limit_speakers
from pii_scrubber import number_tags


def _seg(spk, start, end):
    return {"speaker": spk, "start": start, "end": end, "text": "x"}


def test_limit_speakers_folds_split_label_into_matching_voice():
    segs = [_seg("S01", 0, 60), _seg("S02", 60, 100), _seg("S03", 100, 110)]
    emb = {
        "S01": (np.array([1.0, 0.0]), 60.0),
        "S02": (np.array([0.0, 1.0]), 40.0),
        "S03": (np.array([0.1, 0.9]), 10.0),  # S02's voice, relabelled
    }
    out = limit_speakers(segs, emb, 2)
    assert [s["speaker"] for s in out] == ["S01", "S02", "S02"]


def test_limit_speakers_without_voice_joins_main_speaker():
    segs = [_seg("S01", 0, 60), _seg("S02", 60, 100), _seg("S03", 100, 101)]
    out = limit_speakers(segs, {}, 2)
    assert {s["speaker"] for s in out} == {"S01", "S02"}
    assert out[2]["speaker"] == "S01"


def test_limit_speakers_leaves_small_counts_alone():
    segs = [_seg("S01", 0, 5), _seg("S02", 5, 9)]
    assert limit_speakers(segs, {}, 2) is segs


def test_wav_audio_slices_like_an_array(tmp_path):
    sf = pytest.importorskip("soundfile")
    from moss_windowed import _open_audio, _WavAudio

    data = (np.arange(16000 * 3) % 100 / 100).astype(np.float32)
    path = tmp_path / "a.wav"
    sf.write(path, data, 16000, subtype="FLOAT")
    audio = _open_audio(str(path), 16000)
    assert isinstance(audio, _WavAudio)
    assert len(audio) == len(data)
    np.testing.assert_allclose(audio[16000:16010], data[16000:16010])
    assert len(audio[len(data) - 5:len(data) + 100]) == 5  # clipped at the end
    audio.close()


def _ent(label, start, key):
    return SimpleNamespace(label=label, start=start, text=key,
                           metadata={"normalized_text_hash": key})


def test_number_tags_is_consistent_within_a_recording():
    ids: dict = {}
    first = number_tags("[first_name] met [first_name] in [city].",
                        [_ent("first_name", 0, "a"), _ent("first_name", 17, "b"),
                         _ent("city", 34, "c")], ids)
    second = number_tags("[first_name] again.", [_ent("first_name", 0, "b")], ids)
    assert first == "[first_name_1] met [first_name_2] in [city_1]."
    assert second == "[first_name_2] again."
    assert all("a" != k[1] or v == 1 for k, v in ids.items())


def test_number_tags_falls_back_when_tags_and_entities_disagree():
    masked = "[first_name] and [last_name]"
    assert number_tags(masked, [_ent("first_name", 0, "a")], {}) == masked


def test_ids_csv_matches_with_or_without_extension(tmp_path):
    from batch_processor import _read_ids

    path = tmp_path / "ids.csv"
    path.write_text("filename,participant_id,session_label\nP01_scid.wav,P01,baseline\n")
    ids = _read_ids(str(path))
    assert ids["P01_scid.wav"]["participant_id"] == "P01"
    assert ids["P01_scid"]["session_label"] == "baseline"


def test_append_row_keeps_earlier_rows_and_widens_header(tmp_path):
    from batch_processor import _append_row

    out = tmp_path / "summary.csv"
    _append_row(out, {"filename": "a.wav", "word_count": 10})
    _append_row(out, {"filename": "b.wav", "word_count": 20})
    _append_row(out, {"filename": "c.wav", "word_count": 5, "new_col": 1})
    table = pd.read_csv(out)
    assert list(table["filename"]) == ["a.wav", "b.wav", "c.wav"]
    assert table["new_col"].isna().sum() == 2


def test_stale_scratch_is_cleared_but_live_runs_are_not(tmp_path, monkeypatch):
    import os

    import inference_pipeline as ip

    monkeypatch.setattr(ip, "_SCRATCH_ROOT", tmp_path)
    dead, live = tmp_path / "999999999", tmp_path / str(os.getppid())
    for d in (dead, live):
        d.mkdir()
        (d / "x_16k.wav").write_bytes(b"audio")
    assert ip.clear_stale_scratch() == 1
    assert not dead.exists() and live.exists()


def test_offline_lock_refuses_remote_but_allows_loopback(monkeypatch):
    import socket

    import offline

    monkeypatch.delenv(offline.ALLOW_ENV, raising=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "0")  # restored after the test; lock() sets them
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        assert offline.lock()
        remote = socket.socket()
        with pytest.raises(ConnectionRefusedError, match="offline"):
            remote.connect(("93.184.216.34", 80))
        remote.close()
        local = socket.socket()
        local.connect(server.getsockname())  # this machine: allowed
        local.close()
    finally:
        offline.unlock()
        server.close()


def test_word_count_reports_with_and_without_masked_identifiers():
    from inference_pipeline import InferencePipeline

    text = "My name is [first_name_1] [last_name_1] from [city_1]. It was [REDACTED] fine."
    stats = InferencePipeline._compute_statistics(text, [{"end": 10.0}])
    assert stats["word_count"] == 11          # what the person said
    assert stats["masked_word_count"] == 4
    assert stats["word_count_excluding_masked"] == 7

    plain = InferencePipeline._compute_statistics("no identifiers here at all", [{"end": 2.0}])
    assert plain["word_count"] == plain["word_count_excluding_masked"] == 5
    assert plain["masked_word_count"] == 0


def test_resume_with_nothing_left_is_success_not_failure(tmp_path, monkeypatch, capsys):
    import batch_processor

    out = tmp_path / "summary.csv"
    out.write_text("filename,word_count\na.wav,10\n")
    monkeypatch.setattr(batch_processor, "batch_process", lambda *a, **k: pd.DataFrame())
    batch_processor.main(["-i", str(tmp_path), "-o", str(out), "--resume"])  # no SystemExit
    assert "Nothing left to process" in capsys.readouterr().out


def test_batch_finds_uppercase_extensions_and_skips_macos_stubs(tmp_path):
    from batch_processor import _find_audio_files
    from cw_config import AUDIO_EXTENSIONS

    (tmp_path / "sub").mkdir()
    for name in ("ZOOM0001.WAV", "sub/b.flac", "._ZOOM0001.WAV", ".hidden.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = [p.relative_to(tmp_path).as_posix()
             for p in _find_audio_files(str(tmp_path), list(AUDIO_EXTENSIONS))]
    assert found == ["ZOOM0001.WAV", "sub/b.flac"]
