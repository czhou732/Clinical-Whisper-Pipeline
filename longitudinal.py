#!/usr/bin/env python3
"""Longitudinal tracking — compare acoustic and sentiment features across sessions for the same patient."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("ClinicalWhisper.longitudinal")

# Metrics to track across sessions.
TRACKED_METRICS: list[str] = [
    "vta",
    "pitch_cv",
    "loudness_cv",
    "sentiment_score",
    "word_count",
]

# Slope thresholds for trend classification.
_TREND_THRESHOLD: float = 0.01


def _classify_trend(slope: float) -> str:
    """Classify a linear-regression slope into a human-readable trend label."""
    if slope > _TREND_THRESHOLD:
        return "improving"
    elif slope < -_TREND_THRESHOLD:
        return "declining"
    return "stable"


def _extract_session(data: dict[str, Any], filename: str) -> dict[str, Any]:
    """Pull the tracked metrics out of one analysis JSON payload."""
    stats = data.get("statistics", {})
    sentiment = data.get("overall_sentiment", {})
    acoustics = data.get("overall_acoustics", {})

    return {
        "filename": filename,
        "job_id": data.get("job_id", ""),
        "vta": acoustics.get("vta"),
        "pitch_cv": acoustics.get("pitch_cv"),
        "loudness_cv": acoustics.get("loudness_cv"),
        "sentiment_score": sentiment.get("score"),
        "word_count": stats.get("word_count"),
    }


def _compute_trends(sessions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit a line to each tracked metric across sessions and classify the trend."""
    trends: dict[str, dict[str, Any]] = {}

    for metric in TRACKED_METRICS:
        values = [s.get(metric) for s in sessions]
        # Filter out None / missing values while keeping index alignment
        valid: list[tuple[int, float]] = [
            (i, float(v)) for i, v in enumerate(values) if v is not None
        ]

        if len(valid) < 2:
            trends[metric] = {"trend": "insufficient_data", "slope": None, "values": values}
            continue

        x = np.array([pt[0] for pt in valid], dtype=float)
        y = np.array([pt[1] for pt in valid], dtype=float)

        # Normalize x to [0, 1] to make slope scale-independent
        if x[-1] != x[0]:
            x_norm = (x - x[0]) / (x[-1] - x[0])
        else:
            x_norm = x - x[0]

        coeffs = np.polyfit(x_norm, y, deg=1)
        slope = float(coeffs[0])

        trends[metric] = {
            "trend": _classify_trend(slope),
            "slope": round(slope, 4),
            "values": values,
        }

    return trends


def _build_summary(patient_id: str, n_sessions: int, trends: dict[str, dict[str, Any]]) -> str:
    """Produce a one-paragraph human-readable summary of the longitudinal data."""
    parts: list[str] = [
        f"Patient {patient_id}: {n_sessions} session(s) analyzed.",
    ]

    for metric, info in trends.items():
        label = info["trend"]
        slope = info.get("slope")
        if slope is not None:
            parts.append(f"  {metric}: {label} (slope={slope})")
        else:
            parts.append(f"  {metric}: {label}")

    return "\n".join(parts)


def track_patient(patient_id: str, analysis_dir: str) -> dict[str, Any]:
    """Track a patient's acoustic and sentiment features across sessions.

    Searches *analysis_dir* for JSON files whose names contain *patient_id*
    (glob: ``*{patient_id}*_analysis.json``), sorts by filename, extracts
    key metrics, and fits linear trends.

    Args:
        patient_id:   Identifier to match in filenames (e.g. ``P001``).
        analysis_dir: Directory containing ``*_analysis.json`` files.

    Returns:
        A dict with keys ``patient_id``, ``sessions``, ``trends``, and ``summary``.
    """
    src = Path(analysis_dir).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Analysis directory does not exist: {src}")

    pattern = f"*{patient_id}*_analysis.json"
    matched_files = sorted(src.glob(pattern))

    if not matched_files:
        log.warning("No analysis files matching '%s' in %s", pattern, src)
        return {
            "patient_id": patient_id,
            "sessions": [],
            "trends": {},
            "summary": f"Patient {patient_id}: no matching sessions found.",
        }

    sessions: list[dict[str, Any]] = []
    for jf in matched_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            sessions.append(_extract_session(data, jf.name))
        except Exception as exc:
            log.warning("Skipping %s: %s", jf.name, exc)

    trends = _compute_trends(sessions)
    summary = _build_summary(patient_id, len(sessions), trends)

    log.info("Tracked %d session(s) for patient %s", len(sessions), patient_id)

    return {
        "patient_id": patient_id,
        "sessions": sessions,
        "trends": trends,
        "summary": summary,
    }


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="longitudinal",
        description="ClinicalWhisper longitudinal tracker — compare features across sessions for one patient.",
    )
    parser.add_argument(
        "--patient", "-p",
        required=True,
        help="Patient ID to search for in analysis filenames.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing *_analysis.json files.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    result = track_patient(args.patient, args.input)

    print(f"\n{'=' * 60}")
    print(result["summary"])
    print(f"{'=' * 60}")

    if result["sessions"]:
        print(f"\nSessions ({len(result['sessions'])}):")
        for s in result["sessions"]:
            print(f"  {s['filename']}")
            for m in TRACKED_METRICS:
                val = s.get(m)
                print(f"    {m}: {val}")

        print("\nTrends:")
        for metric, info in result["trends"].items():
            print(f"  {metric}: {info['trend']} (slope={info.get('slope')})")
    else:
        print("\nNo sessions found.")


if __name__ == "__main__":
    main()
