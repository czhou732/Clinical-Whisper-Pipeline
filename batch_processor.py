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

import offline

# Before anything imports huggingface_hub or transformers (see offline.py).
offline.lock()

import pandas as pd  # noqa: E402

from cw_config import load_config, resolve_path  # noqa: E402
from inference_pipeline import InferencePipeline  # noqa: E402

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
    deid = stats.get("deidentification") or {}
    quality = data.get("quality") or {}
    reliability = data.get("score_reliability") or {}
    llm_scores = data.get("llm_clinical_scoring", {})
    acoustics = data.get("overall_acoustics", {})
    timing = data.get("timing_features") or {}
    subject = timing.get("subject_speaker")
    subject_timing = (timing.get("per_speaker") or {}).get(subject, {})
    # Whole-file acoustics mix interviewer and participant; these are the
    # participant's own speech only.
    subject_acoustics = (data.get("speaker_acoustics") or {}).get(subject, {}) or {}
    acoustic_keys = ("vta", "pitch_mean_st", "pitch_cv", "loudness_mean_db",
                     "loudness_cv", "jitter", "shimmer")

    return {
        "filename": audio_filename,
        "participant_id": data.get("participant_id", ""),
        "session_label": data.get("session_label", ""),
        "criterion_score": data.get("criterion_score", ""),
        "analysis_json": analysis_path,
        "status": data.get("status", ""),
        # Codes only; the messages are in the analysis JSON and the app.
        "quality_flags": ";".join(f["code"] for f in quality.get("flags", [])),
        "participant_speech_min": round((quality.get("participant_speech_s") or 0) / 60, 2),
        "word_count": stats.get("word_count", 0),
        # Same count with masked identifiers left out, for analyses that want it.
        "word_count_excluding_masked": stats.get("word_count_excluding_masked"),
        "duration_minutes": round(stats.get("duration_seconds", 0.0) / 60.0, 2),
        "speakers_found": len({s.get("speaker") for s in data.get("segments", [])}),
        "masked_mentions": deid.get("masked_mentions"),
        "distinct_identifiers": deid.get("distinct_identifiers"),
        "hesitancy_score": llm_scores.get("hesitancy_score", None),
        "affect_flatness": llm_scores.get("affect_flatness", None),
        "engagement_level": llm_scores.get("engagement_level", None),
        "elaboration_positive": llm_scores.get("elaboration_positive", None),
        "elaboration_negative": llm_scores.get("elaboration_negative", None),
        "psychomotor_indicators": llm_scores.get("psychomotor_indicators", None),
        # Measured test-retest ICC of each score above, at this file's run count.
        **{f"{k}_icc": v.get("icc") for k, v in reliability.items()},
        # Whole recording, both speakers.
        **{k: acoustics.get(k) for k in acoustic_keys},
        **{f"subject_{k}": subject_acoustics.get(k) for k in acoustic_keys},
        # The subject's speech timing (see timing_features.py).
        **{
            f"subject_{k}": subject_timing.get(k)
            for k in (
                "speech_rate_wps", "pause_proportion", "pause_mean_s", "pause_sd_s",
                "response_latency_median_s", "filler_rate", "talk_time_s",
            )
        },
    }


_ID_COLUMNS = ("participant_id", "session_label", "criterion_score")


def _read_ids(path: str) -> dict[str, dict]:
    """Map filename (with or without extension) -> study identifiers.

    The CSV needs a ``filename`` column plus any of ``participant_id``,
    ``session_label`` and ``criterion_score``.
    """
    table = pd.read_csv(path, dtype=str).fillna("")
    if "filename" not in table.columns:
        raise ValueError(f"{path} needs a 'filename' column (got {list(table.columns)}).")
    ids: dict[str, dict] = {}
    for row in table.to_dict("records"):
        name = Path(row["filename"].strip()).name
        entry = {k: row.get(k, "").strip() for k in _ID_COLUMNS}
        ids[name] = ids[Path(name).stem] = entry
    return ids


def _append_row(out: Path, row: dict) -> None:
    """Add one row to the summary CSV, writing the header on first use.

    Rows are written as files finish, so a run stopped on day two keeps
    everything done before it.
    """
    frame = pd.DataFrame([row])
    if out.exists() and out.stat().st_size:
        existing = pd.read_csv(out, nrows=0).columns.tolist()
        frame = frame.reindex(columns=existing + [c for c in frame.columns if c not in existing])
        if list(frame.columns) != existing:
            # A new column appeared: rewrite with the widened header.
            pd.concat([pd.read_csv(out, dtype=str), frame]).to_csv(out, index=False)
            return
        frame.to_csv(out, mode="a", header=False, index=False)
    else:
        frame.to_csv(out, index=False)


def batch_process(
    input_dir: str,
    output_csv: str,
    config_path: str = "config.yaml",
    transcribe_only: bool = False,
    device: Optional[str] = None,
    ids_csv: Optional[str] = None,
    num_speakers: Optional[int] = None,
    resume: bool = False,
    audio_retention: str = "keep",
) -> pd.DataFrame:
    """Process every audio file in *input_dir* and write a summary CSV.

    Args:
        input_dir:   Directory containing audio files (searched recursively).
        output_csv:  Path for the output CSV file. Rows are appended as files
                     finish.
        config_path: Path to the ClinicalWhisper config.yaml.
        ids_csv:     Optional CSV mapping filename -> participant_id,
                     session_label, criterion_score. Files not listed use their
                     filename (without extension) as the participant ID.
        num_speakers: Known speaker count per recording (2 for an interview).
        resume:      Skip files already listed in *output_csv*.
        audio_retention: ``keep`` (default) leaves the recordings where they
                     are; ``archive`` moves them and ``delete`` removes them
                     after processing, as in the app.

    Returns:
        A :class:`pandas.DataFrame` with one row per file processed in this run.
    """
    if not Path(config_path).exists():
        # No user config: use the shipped defaults (hotwords, windowing).
        # Beside this file from source; the bundle's resources inside the app.
        bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        config_path = str(bundle / "config.example.yaml")
    cfg = load_config(config_path)
    if transcribe_only:
        # Transcript, de-identification, acoustics and timing only — no 5 GB
        # scoring model. The usual need on a compute cluster.
        cfg.setdefault("llm_scoring", {})["enabled"] = False
        cfg["llm_scoring"]["skipped_by_request"] = True
    if device:
        cfg.setdefault("moss", {})["device"] = device
    if num_speakers:
        cfg.setdefault("moss", {})["num_speakers"] = num_speakers
    cfg["audio_retention"] = audio_retention
    scoring = cfg.get("llm_scoring", {}).get("enabled", True)
    extensions: list[str] = cfg.get("audio_extensions", [".m4a", ".mp3", ".wav", ".mp4"])

    root = Path(input_dir).expanduser().resolve()
    audio_files = _find_audio_files(input_dir, extensions)
    if not audio_files:
        log.warning("No audio files found in %s with extensions %s", input_dir, extensions)
        return pd.DataFrame()

    out = Path(output_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    names = {p: str(p.relative_to(root)) for p in audio_files}
    if resume and out.exists() and out.stat().st_size:
        done = set(pd.read_csv(out, usecols=["filename"], dtype=str)["filename"])
        skipped = [p for p in audio_files if names[p] in done]
        audio_files = [p for p in audio_files if names[p] not in done]
        log.info("Resuming: %d file(s) already in %s, %d to go.", len(skipped), out, len(audio_files))
    elif out.exists() and out.stat().st_size and not resume:
        raise FileExistsError(f"{out} already exists. Pass --resume to continue it, or choose a new path.")

    ids = _read_ids(ids_csv) if ids_csv else {}
    log.info("Found %d audio file(s) to process in %s", len(audio_files), input_dir)
    analysis_dir = resolve_path(cfg.get("pipeline", {}).get(
        "analysis_output_folder", cfg.get("output_folder", "./Output")))
    log.info("Summary CSV: %s | per-file results (transcript, JSON): %s", out, analysis_dir)

    pipeline = InferencePipeline(cfg)
    rows: list[dict] = []
    jobs = []
    for audio_path in audio_files:
        meta = ids.get(audio_path.name) or ids.get(audio_path.stem) or {}
        if ids and not meta:
            log.warning("  %s is not in %s; using the filename as participant ID.",
                        audio_path.name, ids_csv)
        jobs.append({
            "job_id": f"batch_{uuid.uuid4().hex[:12]}",
            "file_path": str(audio_path),
            "original_filename": audio_path.name,
            "participant_id": meta.get("participant_id") or audio_path.stem,
            "session_label": meta.get("session_label", ""),
            "criterion_score": meta.get("criterion_score", ""),
        })

    # Files are transcribed in groups of ~2 hours of audio; each group is then
    # scored and written before the next starts, so the transcription and
    # scoring models are never resident at once.
    try:
        for group in pipeline.iter_transcribed(jobs):
            if scoring:
                pipeline.release_transcriber()
            for index, state in group:
                audio_path = audio_files[index]
                if isinstance(state, Exception):
                    log.warning("  ✗ Failed to transcribe %s: %s", audio_path.name, state)
                    continue
                try:
                    analysis_json_path: str = pipeline.score_job(state)
                    row = _extract_row(analysis_json_path, names[audio_path])
                    _append_row(out, row)
                    rows.append(row)
                    log.info("  ✓ %s complete", audio_path.name)
                except Exception as exc:
                    log.warning("  ✗ Failed to process %s: %s", audio_path.name, exc)
            if scoring:
                pipeline.release_scorer()
    finally:
        pipeline.release_all()

    if rows:
        log.info("Summary CSV written to %s (%d new rows)", out, len(rows))
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None, prog: str = "clinicalwhisper-batch") -> None:
    """CLI entry point (also ``ClinicalWhisper --batch`` inside the app)."""
    parser = argparse.ArgumentParser(
        prog=prog,
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
        help="Path to config.yaml (default: config.yaml, else the shipped defaults).",
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Skip LLM clinical scoring: transcript, de-identification, acoustics "
             "and timing features only.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="CSV with a 'filename' column and any of participant_id, session_label, "
             "criterion_score. Unlisted files use their filename as participant ID.",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Speakers per recording, when known (2 for a one-to-one interview). "
             "Extra speaker labels are folded into this many.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: skip files already in the output CSV.",
    )
    parser.add_argument(
        "--audio-retention",
        choices=["keep", "archive", "delete"],
        default="keep",
        help="What to do with each recording after processing (default: keep it "
             "where it is).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Compute device (default: auto — MLX on Apple Silicon, else CUDA, else CPU).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    if not offline.network_allowed():
        offline.require_models(scoring=not args.transcribe_only)

    df = batch_process(
        args.input, args.output, config_path=args.config,
        transcribe_only=args.transcribe_only, device=args.device,
        ids_csv=args.ids, num_speakers=args.speakers, resume=args.resume,
        audio_retention=args.audio_retention,
    )
    if df.empty:
        out = Path(args.output).expanduser()
        if args.resume and out.exists() and out.stat().st_size:
            # Everything was already done: a finished job, not a failure.
            print(f"Nothing left to process: every file is already in {out}.")
            return
        print("No files were processed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessed {len(df)} file(s). Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
