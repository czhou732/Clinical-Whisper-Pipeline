#!/usr/bin/env python3
"""Batch processor for ClinicalWhisper — process a directory of audio files into a summary CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from cw_config import load_config, resolve_path
from inference_pipeline import InferencePipeline

log = logging.getLogger("ClinicalWhisper.batch")


def _find_audio_files(input_dir: str, extensions: list[str]) -> list[Path]:
    """Recursively find all audio files matching the configured extensions."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    # Sort for deterministic ordering
    return sorted(set(files))


def _extract_row(analysis_path: str, audio_filename: str) -> dict:
    """Read an analysis JSON and flatten it into a single CSV-ready dict."""
    with open(analysis_path, "r", encoding="utf-8") as fh:
        data: dict = json.load(fh)

    stats = data.get("statistics", {})
    sentiment = data.get("overall_sentiment", {})
    acoustics = data.get("overall_acoustics", {})

    return {
        "filename": audio_filename,
        "word_count": stats.get("word_count", 0),
        "duration_minutes": round(stats.get("duration_seconds", 0.0) / 60.0, 2),
        "sentiment_score": sentiment.get("score", None),
        "sentiment_label": sentiment.get("label", None),
        "vta": acoustics.get("vta", None),
        "pitch_mean_st": acoustics.get("pitch_mean_st", None),
        "pitch_cv": acoustics.get("pitch_cv", None),
        "loudness_mean_db": acoustics.get("loudness_mean_db", None),
        "loudness_cv": acoustics.get("loudness_cv", None),
        "jitter": acoustics.get("jitter", None),
        "shimmer": acoustics.get("shimmer", None),
    }


def batch_process(
    input_dir: str,
    output_csv: str,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Process every audio file in *input_dir* and write a summary CSV.

    Args:
        input_dir:   Directory containing audio files (searched recursively).
        output_csv:  Path for the output CSV file.
        config_path: Path to the ClinicalWhisper config.yaml.

    Returns:
        A :class:`pandas.DataFrame` with one row per successfully processed file.
    """
    cfg = load_config(config_path)
    extensions: list[str] = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".mp4"])

    audio_files = _find_audio_files(input_dir, extensions)
    if not audio_files:
        log.warning("No audio files found in %s with extensions %s", input_dir, extensions)
        return pd.DataFrame()

    log.info("Found %d audio file(s) in %s", len(audio_files), input_dir)

    pipeline = InferencePipeline(cfg)
    rows: list[dict] = []

    for idx, audio_path in enumerate(audio_files, start=1):
        log.info("[%d/%d] Processing: %s", idx, len(audio_files), audio_path.name)
        try:
            job: dict = {
                "job_id": f"batch_{uuid.uuid4().hex[:12]}",
                "file_path": str(audio_path),
                "original_filename": audio_path.name,
            }
            analysis_json_path: str = pipeline.process_job(job)
            row = _extract_row(analysis_json_path, audio_path.name)
            rows.append(row)
            log.info("  ✓ %s complete", audio_path.name)
        except Exception as exc:
            log.warning("  ✗ Failed to process %s: %s", audio_path.name, exc)

    df = pd.DataFrame(rows)

    if not df.empty:
        out = Path(output_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out), index=False)
        log.info("Summary CSV written to %s (%d rows)", out, len(df))

    return df


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="batch_processor",
        description="ClinicalWhisper batch processor — run the full pipeline on a directory of audio files.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing audio files to process.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the output summary CSV.",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    df = batch_process(args.input, args.output, config_path=args.config)
    if df.empty:
        print("No files were processed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessed {len(df)} file(s). Summary saved to: {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
