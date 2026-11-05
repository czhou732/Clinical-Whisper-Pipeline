"""Silencing masked names in the audio."""

import numpy as np
import pytest

import audio_deid as ad

sf = pytest.importorskip("soundfile")


def test_masked_words_are_found():
    raw = "So we had Maria Lopez, we had Carlos. Okay."
    masked = "So we had [first_name_1] [last_name_1], we had [first_name_2]. Okay."
    assert ad.masked_word_indices(raw, masked) == [3, 4, 7]
    assert ad.masked_word_indices("nothing here", "nothing here") == []
    # A tag that can't be lined up silences the whole segment's words.
    assert ad.masked_word_indices("a b c", "[REDACTED]") == [0, 1, 2]


def test_spans_merge_and_write_silences(tmp_path):
    assert ad._merge([(1.0, 2.0), (1.5, 3.0), (5.0, 6.0)]) == [[1.0, 3.0], [5.0, 6.0]]
    wav = tmp_path / "a.wav"
    sf.write(wav, np.full(16000 * 4, 0.5, dtype=np.float32), 16000)
    out = ad.write(str(wav), [[1.0, 2.0]], tmp_path / "out" / "a_names_silenced.wav")
    audio, _ = sf.read(out, dtype="float32")
    assert np.all(audio[16000:32000] == 0) and np.all(audio[:16000] > 0.4)


def test_without_aligner_whole_segments_are_silenced(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.full(16000 * 10, 0.5, dtype=np.float32), 16000)
    raw = [{"start": 0, "end": 3, "text": "My name is Maria."},
           {"start": 4, "end": 6, "text": "Nothing here."}]
    masked = [{"start": 0, "end": 3, "text": "My name is [first_name_1]."},
              {"start": 4, "end": 6, "text": "Nothing here."}]
    found = ad.silence_spans(raw, masked, str(wav), use_aligner=False)
    assert found["spans"] == [[0.0, 3.0]] and found["segments_silenced_whole"] == 1
