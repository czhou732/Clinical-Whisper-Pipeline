#!/usr/bin/env python3
"""Export ClinicalWhisper analysis results to CSV for statistical analysis (R/SPSS compatible)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("ClinicalWhisper.export")

# Column order for the output CSV — matches typical SPSS/R import expectations.
COLUMNS: list[str] = [
    "job_id",
    "original_filename",
    "model",
    "word_count",
    "character_count",
    "sentence_count",
    "duration_minutes",
    "overall_sentiment_label",
    "overall_sentiment_score",
    "overall_polarity",
    "overall_subjectivity",
    "pitch_mean_st",
    "pitch_cv",
    "loudness_mean_db",
    "loudness_cv",
    "jitter",
    "shimmer",
    "vta",
    "tones_list",
    "critical_moments_count",
    "num_speakers",
]


def _flatten_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single *_analysis.json payload into a one-level dict."""
    stats = data.get("statistics", {})
    sentiment = data.get("overall_sentiment", {})
    acoustics = data.get("overall_acoustics", {})
    source = data.get("source_audio", {})

    duration_sec = stats.get("duration_seconds", 0.0)

    # Determine number of unique speakers
    speaker_acoustics = data.get("speaker_acoustics", {})
    speaker_sentiments = data.get("speaker_sentiments", {})
    all_speakers = set(speaker_acoustics.keys()) | set(speaker_sentiments.keys())
    # Fall back to counting distinct speakers in segments
    if not all_speakers:
        all_speakers = {
            seg.get("speaker", "Speaker 1")
            for seg in data.get("segments", [])
            if seg.get("speaker")
        }

    tones: list[str] = data.get("tones", [])

    return {
        "job_id": data.get("job_id", ""),
        "original_filename": source.get("original_filename", ""),
        "model": data.get("model", ""),
        "word_count": stats.get("word_count", 0),
        "character_count": stats.get("character_count", 0),
        "sentence_count": stats.get("sentence_count", 0),
        "duration_minutes": round(duration_sec / 60.0, 2),
        "overall_sentiment_label": sentiment.get("label", ""),
        "overall_sentiment_score": sentiment.get("score", ""),
        "overall_polarity": sentiment.get("polarity", ""),
        "overall_subjectivity": sentiment.get("subjectivity", ""),
        "pitch_mean_st": acoustics.get("pitch_mean_st", ""),
        "pitch_cv": acoustics.get("pitch_cv", ""),
        "loudness_mean_db": acoustics.get("loudness_mean_db", ""),
        "loudness_cv": acoustics.get("loudness_cv", ""),
        "jitter": acoustics.get("jitter", ""),
        "shimmer": acoustics.get("shimmer", ""),
        "vta": acoustics.get("vta", ""),
        "tones_list": ", ".join(tones) if tones else "",
        "critical_moments_count": len(data.get("critical_moments", [])),
        "num_speakers": len(all_speakers) if all_speakers else 1,
    }


def export_to_csv(analysis_dir: str, output_path: str) -> str:
    """Read all ``*_analysis.json`` files from *analysis_dir* and write a flat CSV.

    Args:
        analysis_dir: Directory containing ClinicalWhisper analysis JSON files.
        output_path:  Destination path for the CSV.

    Returns:
        The absolute path of the written CSV file.

    Raises:
        FileNotFoundError: If *analysis_dir* does not exist.
    """
    src = Path(analysis_dir).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Analysis directory does not exist: {src}")

    json_files = sorted(src.glob("*_analysis.json"))
    if not json_files:
        log.warning("No *_analysis.json files found in %s", src)

    rows: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            rows.append(_flatten_analysis(data))
        except Exception as exc:
            log.warning("Skipping %s: %s", jf.name, exc)

    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log.info("Exported %d record(s) to %s", len(rows), dest)
    return str(dest)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="export_features",
        description="Export ClinicalWhisper analysis JSONs to a flat CSV for R / SPSS.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing *_analysis.json files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    result_path = export_to_csv(args.input, args.output)
    print(f"Features exported to: {result_path}")


if __name__ == "__main__":
    main()
