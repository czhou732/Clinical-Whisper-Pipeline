"""Time the full pipeline on one file, broken down by stage.

    uv run python evals/bench_pipeline.py recording.mp3 --out /tmp/cw_bench

Writes the analysis JSON to ``--out`` and prints seconds per stage, total, and
the real-time factor, so speed regressions show up as numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cw_config import load_config  # noqa: E402
from inference_pipeline import InferencePipeline  # noqa: E402
from moss_diarizer import _audio_duration  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("--out", required=True, help="directory for analysis output")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.example.yaml"),
                    help="config to benchmark (default: the shipped config.example.yaml)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["output_folder"] = cfg["processed_folder"] = str(out)
    cfg["pipeline"]["analysis_output_folder"] = str(out)
    cfg["pipeline"]["secure_storage_folder"] = str(out)

    stages: dict[str, float] = {}
    current = [None, time.perf_counter()]

    def progress(stage, fraction=None, detail=""):
        now = time.perf_counter()
        if stage != current[0]:
            if current[0] is not None:
                stages[current[0]] = stages.get(current[0], 0.0) + now - current[1]
            current[0], current[1] = stage, now

    pipeline = InferencePipeline(cfg, progress_cb=progress)
    job = {"job_id": f"bench_{uuid.uuid4().hex[:8]}", "file_path": args.audio,
           "original_filename": Path(args.audio).name}

    t0 = time.perf_counter()
    state = pipeline.transcribe_job(job)
    t1 = time.perf_counter()
    pipeline.release_transcriber()
    analysis = pipeline.score_job(state)
    t2 = time.perf_counter()
    progress("__end__")
    pipeline.release_all()

    audio_s = _audio_duration(args.audio) or (state["segments"][-1]["end"] if state["segments"] else 0)
    print(json.dumps({
        "audio_seconds": round(audio_s, 1),
        "transcribe_half_s": round(t1 - t0, 1),
        "score_half_s": round(t2 - t1, 1),
        "total_s": round(t2 - t0, 1),
        "x_realtime": round(audio_s / (t2 - t0), 1) if t2 > t0 else None,
        "stages_s": {k: round(v, 1) for k, v in stages.items()},
        "segments": len(state["segments"]),
        "speakers": len({s["speaker"] for s in state["segments"]}),
        "warnings": state.get("warnings", []),
        "analysis": str(analysis),
    }, indent=1))


if __name__ == "__main__":
    main()
