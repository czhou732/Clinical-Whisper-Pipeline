"""Window planning, speaker linking and merging — no model required."""

import numpy as np
import pytest

from moss_chunking import (
    FRAME_SECONDS,
    consolidate_speakers,
    Window,
    link_speakers,
    merge_windows,
    plan_windows,
    speaker_embeddings,
)

SR = 16000


@pytest.mark.parametrize("seconds", [121, 300, 1725.8, 4057.2])
def test_windows_are_equal_length_cover_the_file_and_overlap(seconds):
    n = int(seconds * SR)
    windows = plan_windows(n, SR, window_s=120, overlap_s=15)
    assert len({w.n_samples for w in windows}) == 1
    assert windows[0].start_sample == 0
    assert windows[-1].start_sample + windows[-1].n_samples == n
    for a, b in zip(windows, windows[1:]):
        assert a.end - b.start >= 15 - 1e-6


def test_short_audio_is_one_window():
    windows = plan_windows(90 * SR, SR, window_s=120, overlap_s=15)
    assert len(windows) == 1 and windows[0].n_samples == 90 * SR


def _two_windows():
    return [Window(0, 0, 100 * SR, SR), Window(1, 80 * SR, 100 * SR, SR)]


def test_overlap_agreement_links_swapped_labels():
    windows = _two_windows()
    # Window 1 numbers the same two people the other way round.
    segs = [
        [{"start": 0, "end": 85, "speaker": "S01", "text": "a"},
         {"start": 85, "end": 100, "speaker": "S02", "text": "b"}],
        [{"start": 0, "end": 5, "speaker": "S02", "text": "a"},
         {"start": 5, "end": 20, "speaker": "S01", "text": "b"},
         {"start": 20, "end": 100, "speaker": "S02", "text": "c"}],
    ]
    emb = [{"S01": (np.ones(4), 1.0), "S02": (-np.ones(4), 1.0)}] * 2
    maps = link_speakers(windows, segs, emb, similarity_threshold=0.99)
    assert maps[0] == {"S01": "S01", "S02": "S02"}
    assert maps[1] == {"S02": "S01", "S01": "S02"}


def test_voice_similarity_links_speaker_absent_from_overlap():
    windows = _two_windows()
    segs = [
        [{"start": 0, "end": 40, "speaker": "S01", "text": "a"},
         {"start": 40, "end": 100, "speaker": "S02", "text": "b"}],
        # Only S02-of-window-0 talks in the overlap; the other voice returns later.
        [{"start": 0, "end": 30, "speaker": "S01", "text": "b"},
         {"start": 30, "end": 100, "speaker": "S02", "text": "a"}],
    ]
    voice_a, voice_b = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    emb = [
        {"S01": (voice_a, 40.0), "S02": (voice_b, 60.0)},
        {"S01": (voice_b, 30.0), "S02": (voice_a, 70.0)},
    ]
    maps = link_speakers(windows, segs, emb, similarity_threshold=0.5)
    assert maps[1] == {"S01": "S02", "S02": "S01"}


def test_unrecognised_voice_becomes_new_speaker():
    windows = _two_windows()
    segs = [
        [{"start": 0, "end": 100, "speaker": "S01", "text": "a"}],
        [{"start": 0, "end": 30, "speaker": "S01", "text": "a"},
         {"start": 30, "end": 100, "speaker": "S02", "text": "new"}],
    ]
    emb = [
        {"S01": (np.array([1.0, 0, 0]), 100.0)},
        {"S01": (np.array([1.0, 0, 0]), 30.0), "S02": (np.array([0, 0, 1.0]), 70.0)},
    ]
    maps = link_speakers(windows, segs, emb, similarity_threshold=0.5)
    assert maps[1] == {"S01": "S01", "S02": "S02"}


def test_merge_splits_overlap_at_midpoint_without_duplicates():
    windows = _two_windows()  # overlap 80-100s, cut at 90s
    segs = [
        [{"start": 70, "end": 85, "speaker": "S01", "text": "keep-from-0"},
         {"start": 92, "end": 98, "speaker": "S01", "text": "drop-dup"}],
        [{"start": 0, "end": 5, "speaker": "S01", "text": "drop-dup-early"},
         {"start": 12, "end": 18, "speaker": "S01", "text": "keep-from-1"}],
    ]
    maps = [{"S01": "S01"}, {"S01": "S01"}]
    merged = merge_windows(windows, segs, maps)
    assert [s["text"] for s in merged] == ["keep-from-0", "keep-from-1"]
    assert merged[1]["start"] == 92.0


def test_speaker_embeddings_average_over_that_speakers_frames():
    feats = np.zeros((int(10 / FRAME_SECONDS), 2))
    feats[: int(5 / FRAME_SECONDS), 0] = 1.0
    feats[int(5 / FRAME_SECONDS):, 1] = 1.0
    segs = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 5, "end": 10, "speaker": "B"}]
    emb = speaker_embeddings(segs, feats)
    assert np.allclose(emb["A"][0], [1, 0]) and np.allclose(emb["B"][0], [0, 1])
    assert emb["A"][1] == pytest.approx(5.0, abs=0.1)


def _diarizer_with_config(batch_size):
    from types import SimpleNamespace

    from moss_diarizer import MOSSDiarizer

    d = object.__new__(MOSSDiarizer)
    d.batch_size = batch_size
    text = SimpleNamespace(
        num_hidden_layers=28, num_key_value_heads=8, head_dim=128,
        hidden_size=1024, num_attention_heads=16,
    )
    d._mlx = SimpleNamespace(text_config=vars(text))
    d.backend = "mlx"
    return d


def test_batch_limit_fits_total_and_available_memory(monkeypatch):
    import moss_windowed

    fp16_per_seq = (3814 + 2850) * 28 * 2 * 8 * 128 * 2  # ~0.76 GB
    d = _diarizer_with_config(batch_size=32)
    cases = [
        # (total GB, available GB, expected batch)
        (32, None, int(8e9 // fp16_per_seq)),            # quarter of RAM
        (32, 4, int(2.4e9 // fp16_per_seq)),             # busy machine: 60% of free
        (8, 8, int(2e9 // fp16_per_seq)),                # small Mac
        (256, 256, 32),                                  # capped by batch_size
        (1, 0.1, 1),                                     # never below one
    ]
    for total, avail, expected in cases:
        monkeypatch.setattr(
            moss_windowed, "_memory_bytes",
            lambda t=total, a=avail: (t * 1e9, None if a is None else a * 1e9),
        )
        assert d._batch_limit(3814, 2850) == expected, (total, avail)


def test_consolidate_merges_split_speaker_but_not_co_occurring_ones():
    a, b, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    # Voice "a" is S01 in windows 0-1 and wrongly became S03 in windows 2-3.
    # Voice "c" (S02) is very similar to nothing; voice "b" (S04) resembles "a"
    # but talks in the same window as S01, so it must stay distinct.
    emb = [
        {"x": (a, 50.0), "y": (c, 50.0)},
        {"x": (a, 50.0)},
        {"x": (a * 0.9 + b * 0.1, 60.0), "y": (c, 40.0)},
        {"x": (a, 30.0), "z": (a * 0.8 + b * 0.2, 30.0)},
    ]
    maps = [
        {"x": "S01", "y": "S02"},
        {"x": "S01"},
        {"x": "S03", "y": "S02"},
        {"x": "S03", "z": "S04"},
    ]
    out = consolidate_speakers(emb, maps, merge_threshold=0.75)
    assert out[2]["x"] == out[0]["x"] == "S01"
    assert out[3]["x"] == "S01"
    assert out[3]["z"] not in {"S01", "S02"}
    assert sorted({g for m in out for g in m.values()}) == ["S01", "S02", "S03"]


def test_silence_detection():
    from moss_chunking import is_silent

    quiet = np.zeros(SR * 60, dtype=np.float32)
    assert is_silent(quiet, SR)
    speech = quiet.copy()
    speech[SR * 10:SR * 13] = 0.1 * np.sin(np.linspace(0, 2000, SR * 3))  # ~-23 dBFS for 3 s
    assert not is_silent(speech, SR)


def test_degenerate_decode_detection():
    from moss_chunking import looks_degenerate

    loop = [{"text": "and then we went to the"} for _ in range(4)]
    assert looks_degenerate(loop, tokens=500, budget=4096)
    backchannels = [{"text": "Yeah."} for _ in range(6)]
    assert not looks_degenerate(backchannels, tokens=500, budget=4096)
    assert looks_degenerate([{"text": "fine"}], tokens=4096, budget=4096)


def test_uncertain_speakers_flagged():
    from moss_chunking import flag_uncertain_speakers

    segs = [{"speaker": "S01", "start": 0, "end": 60}, {"speaker": "S02", "start": 60, "end": 63}]
    assert flag_uncertain_speakers(segs) == ["S02"]
    assert segs[1]["speaker_uncertain"] and "speaker_uncertain" not in segs[0]


def test_split_threshold_joins_one_voice_given_two_labels_in_a_window():
    windows = [Window(0, 0, 100 * SR, SR)]
    segs = [[
        {"speaker": "S01", "start": 0, "end": 30, "text": "a"},
        {"speaker": "S02", "start": 30, "end": 60, "text": "b"},   # same person, relabelled
        {"speaker": "S03", "start": 60, "end": 90, "text": "c"},   # someone else
    ]]
    v = np.array([1.0, 0.1, 0]); other = np.array([0, 0, 1.0])
    emb = [{"S01": (v, 30.0), "S02": (v * 1.01, 30.0), "S03": (other, 30.0)}]
    kept_apart = link_speakers(windows, segs, emb, 0.5, center=False)[0]
    assert len(set(kept_apart.values())) == 3
    joined = link_speakers(windows, segs, emb, 0.5, center=False, split_threshold=0.8)[0]
    assert joined["S01"] == joined["S02"] != joined["S03"]
