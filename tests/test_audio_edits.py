"""Leaving parts of a recording out, and keeping original timestamps."""

import numpy as np
import pytest

from audio_edits import GAP_S, Edits, TimeMap, apply, parse_ranges, parse_time

SR = 100  # small rate keeps the arrays readable


@pytest.mark.parametrize("text,sec", [("45", 45), ("12:30", 750), ("1:02:03", 3723),
                                      ("2.5", 2.5), ("", None), (90, 90.0)])
def test_parse_time(text, sec):
    assert parse_time(text) == sec


def test_parse_time_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_time("ten minutes")


def test_parse_ranges():
    assert parse_ranges("02:30-05:00, 1:10:00 – 1:12:00") == [(150, 300), (4200, 4320)]
    with pytest.raises(ValueError):
        parse_ranges("02:30")


def test_keep_merges_overlaps_and_respects_start_end():
    e = Edits(start=10, end=100, skip=((20, 30), (25, 40), (90, 120)))
    assert e.keep() == [(10, 20), (40, 90)]
    assert Edits(skip=((0, 5),)).keep() == [(5, None)]


def test_invalid_edits_refused():
    with pytest.raises(ValueError):
        Edits.from_dict({"start": "5:00", "end": "4:00"})
    with pytest.raises(ValueError):
        Edits.from_dict({"skip": "05:00-04:00"})
    assert Edits.from_dict({"start": "", "end": "", "skip": ""}) is None


def _stream(seconds, block=37):
    """A ramp where each sample's value is its original time, in odd-sized blocks."""
    x = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return [x[i:i + block] for i in range(0, x.size, block)]


def test_apply_keeps_only_marked_audio_with_gaps():
    edits = Edits(start=1, end=9, skip=((3, 5),))
    out = np.concatenate(list(apply(_stream(10), edits, SR)))
    gap = int(GAP_S * SR)
    first, rest = out[:2 * SR], out[2 * SR:]
    assert first[0] == pytest.approx(1.0) and first[-1] == pytest.approx(2.99)
    assert not rest[:gap].any()
    second = rest[gap:]
    assert second.size == 4 * SR
    assert second[0] == pytest.approx(5.0) and second[-1] == pytest.approx(8.99)


def test_time_map_returns_original_times():
    tm = TimeMap.for_edits(Edits(start=1, end=9, skip=((3, 5),)))
    assert tm.to_original(0.0) == pytest.approx(1.0)
    assert tm.to_original(1.5) == pytest.approx(2.5)
    assert tm.to_original(2.2) == pytest.approx(3.0)   # inside the inserted silence
    assert tm.to_original(2.2, after_cut=True) == pytest.approx(5.0)
    assert tm.to_original(2.5 + 1.0) == pytest.approx(6.0)
    seg = tm.segments([{"start": 0.5, "end": 3.0, "speaker": "S01"}])[0]
    assert (seg["start"], seg["end"]) == (1.5, 5.5)


def test_mapped_audio_and_times_agree():
    """A sample's edited position, mapped back, is its original time."""
    edits = Edits(start=2, skip=((4, 7),))
    out = np.concatenate(list(apply(_stream(12), edits, SR)))
    tm = TimeMap.for_edits(edits)
    for i in (0, 150, 199, 250, 400, out.size - 1):
        if out[i] == 0 and i > 0:
            continue  # inserted silence
        assert tm.to_original(i / SR) == pytest.approx(float(out[i]), abs=1 / SR)


def test_window_edits_are_checked_per_file():
    import gui_server

    assert gui_server._checked_edits(None, 2) == [None, None]
    out = gui_server._checked_edits([{"start": "1:00", "skip": "02:00-03:00"}, None], 2)
    assert out[0] == {"start": 60.0, "end": None, "skip": [[120.0, 180.0]]}
    assert out[1] is None
    with pytest.raises(ValueError):
        gui_server._checked_edits([{"start": "5:00", "end": "1:00"}], 1)
    with pytest.raises(ValueError):
        gui_server._checked_edits([None], 2)


def test_preview_serves_only_registered_files(tmp_path):
    from fastapi.testclient import TestClient

    import gui_server

    f = tmp_path / "a.wav"
    f.write_bytes(b"RIFF....")
    token = gui_server.register_preview(f)
    api = TestClient(gui_server.app, base_url="http://127.0.0.1")
    assert api.get(f"/api/preview/{token}").content == b"RIFF...."
    assert api.get("/api/preview/not-a-token").status_code == 404


def test_ids_csv_can_carry_edits(tmp_path):
    from batch_processor import _read_ids

    csv = tmp_path / "ids.csv"
    csv.write_text("filename,participant_id,start,end,skip\n"
                   "a.wav,P01,0:30,,10:00-12:00\nb.wav,P02,,,\n")
    ids = _read_ids(str(csv))
    assert ids["a.wav"]["audio_edits"] == {"start": 30.0, "end": None, "skip": [[600.0, 720.0]]}
    assert ids["b"]["audio_edits"] is None
