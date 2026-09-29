"""Turn the AMI corpus (Hugging Face ``edinburghcstr/ami``) into test meetings.

The Hugging Face release stores each meeting as utterances — audio clip,
begin/end time, speaker ID and a human transcript. This rebuilds, per meeting:

* ``<meeting>.wav`` — 16 kHz mono, each utterance placed at its begin time.
  Utterances come from one single distant microphone ("sdm"), so overlapping
  speech is the same signal and simply overwrites. Stretches no one annotated
  become silence, which makes the task slightly easier than a raw recording —
  state that wherever these numbers are reported.
* ``<meeting>.ref.json`` — ``[{"speaker", "start", "end", "text"}]``, the human
  reference, sorted by start time.

    uv run python evals/ground_truth/ami_prepare.py ~/Developer/datasets/ami/prepared \\
        ~/Developer/datasets/ami/sdm_test_0.parquet ~/Developer/datasets/ami/sdm_test_1.parquet \\
        --meetings EN2002a EN2002b EN2002c ES2004a ES2004b

A meeting's utterances can be split across parquet shards (EN2002c is 839 in
shard 0 and 630 in shard 1, interleaved in time), so always pass every shard a
meeting touches — and exclude the last meeting of the last shard, which may
continue in a shard not downloaded.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

SAMPLE_RATE = 16000


def _to_mono_16k(data: np.ndarray, rate: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate}")
    return data.astype(np.float32)


def prepare(parquets: list[Path], out_dir: Path, meetings: Optional[list[str]] = None) -> list[str]:
    table = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)
    if meetings:
        table = table[table["meeting_id"].isin(meetings)]
    out_dir.mkdir(parents=True, exist_ok=True)
    meetings = []
    for meeting, rows in table.groupby("meeting_id"):
        rows = rows.sort_values("begin_time")
        length = int(np.ceil(rows["end_time"].max() * SAMPLE_RATE)) + SAMPLE_RATE
        audio = np.zeros(length, dtype=np.float32)
        for _, row in rows.iterrows():
            clip, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
            clip = _to_mono_16k(clip, rate)
            start = int(round(row["begin_time"] * SAMPLE_RATE))
            audio[start:start + len(clip)] = clip[: max(0, length - start)]
        sf.write(out_dir / f"{meeting}.wav", audio, SAMPLE_RATE)
        reference = [
            {
                "speaker": row["speaker_id"],
                "start": round(float(row["begin_time"]), 2),
                "end": round(float(row["end_time"]), 2),
                "text": row["text"],
            }
            for _, row in rows.iterrows()
        ]
        (out_dir / f"{meeting}.ref.json").write_text(json.dumps(reference, indent=1))
        meetings.append(meeting)
        print(f"{meeting}: {length / SAMPLE_RATE / 60:.1f} min, {len(reference)} utterances, "
              f"{rows['speaker_id'].nunique()} speakers")
    return meetings


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("parquets", type=Path, nargs="+")
    ap.add_argument("--meetings", nargs="*", help="only these (complete) meetings")
    args = ap.parse_args()
    prepare([p.expanduser() for p in args.parquets], args.out_dir.expanduser(), args.meetings)
