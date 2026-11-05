"""Benchmark MOSS transcription backends: speed and agreement.

Runs the same file through the PyTorch decoder and/or the MLX decoder and
reports wall time, real-time factor, and word-level agreement with the first
backend listed. Transcripts are written to ``--out`` for inspection.

    uv run python evals/bench_transcription.py clip.wav --backends torch,mlx
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moss_diarizer import MOSSDiarizer, _audio_duration  # noqa: E402


def run(path: str, spec: str) -> dict:
    """``spec`` is ``torch`` or ``mlx[:dtype]``, e.g. ``mlx:float32``."""
    backend, _, dtype = spec.partition(":")
    kwargs = {"mlx_dtype": dtype} if dtype else {}
    d = MOSSDiarizer(backend=backend, **kwargs)
    t = time.perf_counter()
    segments = d.process_file(path)
    elapsed = time.perf_counter() - t
    return {"seconds": elapsed, "segments": segments}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("--backends", default="torch,mlx")
    ap.add_argument("--out", default=None, help="JSON file for transcripts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    duration = _audio_duration(args.audio) or 0.0
    results = {b: run(args.audio, b) for b in args.backends.split(",")}

    ref_words = None
    for name, r in results.items():
        words = " ".join(s["text"] for s in r["segments"]).split()
        ref_words = ref_words if ref_words is not None else words
        agree = difflib.SequenceMatcher(None, ref_words, words, autojunk=False).ratio()
        segs = r["segments"]
        print(json.dumps({
            "backend": name,
            "seconds": round(r["seconds"], 1),
            "audio_seconds": round(duration, 1),
            "x_realtime": round(duration / r["seconds"], 1) if r["seconds"] else None,
            "segments": len(segs),
            "speakers": len({s["speaker"] for s in segs}),
            "last_end": segs[-1]["end"] if segs else None,
            "words": len(words),
            "word_agreement": round(agree, 4),
        }))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {k: v["segments"] for k, v in results.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
