"""AMI meetings with a late joiner, for testing speaker labels in groups.

MOSS labels speakers per 5-minute window. When someone first speaks partway
through a window, MOSS can give them the label of a person already talking
in that window, and the linker then carries the merge through the file. To
test that, this rebuilds AMI test meetings with one participant's
utterances removed before ``--join`` seconds, so they join mid-window
(windows start at 0, 270, 540 ... s; 420 s is mid-window).

The joiner is the participant with the most speech after the join time, so
there is plenty to label. Overlapping speech in AMI's single distant
microphone is one signal, so removing an utterance can also silence a little
of someone else's overlapping speech; the reference keeps every remaining
utterance as annotated.

    python evals/ground_truth/late_joiner_prepare.py ~/Developer/datasets/ami/late_joiner \\
        ~/Developer/datasets/ami/sdm_test_0.parquet ~/Developer/datasets/ami/sdm_test_1.parquet \\
        --meetings EN2002a EN2002b EN2002c ES2004a ES2004b
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from ami_prepare import SAMPLE_RATE, _to_mono_16k


def prepare(parquets: list[Path], out_dir: Path, meetings: list[str], join_s: float) -> None:
    table = pd.concat([pq.read_table(p).to_pandas() for p in parquets], ignore_index=True)
    table = table[table["meeting_id"].isin(meetings)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for meeting, rows in table.groupby("meeting_id"):
        rows = rows.sort_values("begin_time")
        after = rows[rows["begin_time"] >= join_s]
        talk = (after["end_time"] - after["begin_time"]).groupby(after["speaker_id"]).sum()
        joiner = talk.idxmax()
        keep = rows[~((rows["speaker_id"] == joiner) & (rows["begin_time"] < join_s))]
        length = int(np.ceil(keep["end_time"].max() * SAMPLE_RATE)) + SAMPLE_RATE
        audio = np.zeros(length, dtype=np.float32)
        for _, row in keep.iterrows():
            clip, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
            clip = _to_mono_16k(clip, rate)
            start = int(round(row["begin_time"] * SAMPLE_RATE))
            audio[start:start + len(clip)] = clip[: max(0, length - start)]
        name = f"{meeting}_late"
        sf.write(out_dir / f"{name}.wav", audio, SAMPLE_RATE)
        reference = [{"speaker": r["speaker_id"], "start": round(float(r["begin_time"]), 2),
                      "end": round(float(r["end_time"]), 2), "text": r["text"]} for _, r in keep.iterrows()]
        (out_dir / f"{name}.ref.json").write_text(json.dumps(reference, indent=1))
        first = min(r["start"] for r in reference if r["speaker"] == joiner)
        (out_dir / f"{name}.joiner.json").write_text(json.dumps({"speaker": joiner, "first_speech_s": first}))
        print(f"{name}: {length / SAMPLE_RATE / 60:.1f} min, joiner {joiner} first speaks at {first:.0f} s, "
              f"{talk[joiner] / 60:.1f} min of their speech after joining")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("parquets", type=Path, nargs="+")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--join", type=float, default=420.0)
    args = ap.parse_args()
    prepare([p.expanduser() for p in args.parquets], args.out_dir.expanduser(), args.meetings, args.join)
