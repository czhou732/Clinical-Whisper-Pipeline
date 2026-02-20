#!/usr/bin/env python3
"""Inference worker pipeline for ClinicalWhisper async processing."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import whisper
from textblob import TextBlob

from cw_config import resolve_path
from sentiment_analyzer import analyze_per_speaker, analyze_sentiment

log = logging.getLogger("ClinicalWhisper")


def _polarity_to_score(polarity: float) -> int:
    """Map TextBlob polarity (-1..1) to a 1..10 score."""
    return max(1, min(10, round(polarity * 4.5 + 5.5)))


def _segment_sentiment_score(text: str) -> int:
    if not text.strip():
        return 5
    pol = TextBlob(text).sentiment.polarity
    return _polarity_to_score(pol)


class InferencePipeline:
    """Loads models once and processes queued jobs."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("model", "medium.en")
        log.info("Loading Whisper model '%s'...", self.model_name)
        self.model = whisper.load_model(self.model_name)
        self._diarizer = None

    def _get_diarizer(self):
        if self._diarizer is not None:
            return self._diarizer

        diar_cfg = self.cfg.get("diarization", {})
        if not diar_cfg.get("enabled", False):
            return None

        try:
            from speaker_diarizer import SpeakerDiarizer

            self._diarizer = SpeakerDiarizer(
                hf_token=diar_cfg.get("hf_token"),
                min_speakers=diar_cfg.get("min_speakers", 2),
                max_speakers=diar_cfg.get("max_speakers", 6),
            )
            return self._diarizer
        except Exception as exc:
            log.warning("Diarization unavailable, continuing without it: %s", exc)
            return None

    def _build_segments(
        self, whisper_result: dict, merged_segments: Optional[list[dict]]
    ) -> list[dict]:
        segments: list[dict] = []

        if merged_segments:
            source_segments = merged_segments
        else:
            source_segments = [
                {
                    "start": s.get("start", 0.0),
                    "end": s.get("end", 0.0),
                    "speaker": "Speaker 1",
                    "text": (s.get("text") or "").strip(),
                }
                for s in whisper_result.get("segments", [])
            ]

        for seg in source_segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "speaker": seg.get("speaker", "Speaker 1"),
                    "text": text,
                    "sentiment": _segment_sentiment_score(text),
                }
            )

        return segments

    @staticmethod
    def _compute_statistics(transcript: str, segments: list[dict]) -> dict:
        words = transcript.split()
        sentence_count = sum(1 for c in transcript if c in ".!?")
        duration_seconds = 0.0
        if segments:
            duration_seconds = max(float(seg["end"]) for seg in segments)

        return {
            "word_count": len(words),
            "character_count": len(transcript),
            "sentence_count": sentence_count,
            "estimated_minutes": round(len(words) / 150.0, 2),
            "duration_seconds": round(duration_seconds, 2),
        }

    def _archive_audio(self, job_id: str, source_path: Path, original_filename: str) -> str:
        processed_dir = Path(resolve_path(self.cfg.get("processed_folder", "./Processed")))
        processed_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(original_filename).name
        destination = processed_dir / f"{job_id}_{safe_name}"
        shutil.move(str(source_path), str(destination))
        return str(destination)

    def process_job(self, job: dict) -> str:
        """
        Process one queue job and write `[job_id]_analysis.json`.

        Returns:
            Path to the JSON output file.
        """
        job_id = job["job_id"]
        file_path = Path(job["file_path"]).expanduser()
        original_filename = job.get("original_filename", file_path.name)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        log.info("Job %s: transcribing %s", job_id, file_path.name)
        whisper_result = self.model.transcribe(str(file_path))
        transcript = (whisper_result.get("text") or "").strip()

        diarizer = self._get_diarizer()
        merged_segments = None
        speaker_texts = None
        if diarizer:
            try:
                diar_result = diarizer.diarize(str(file_path))
                merged_segments = diarizer.merge_with_transcript(whisper_result, diar_result)
                speaker_texts = diarizer.group_by_speaker(merged_segments)
            except Exception as exc:
                log.warning("Job %s: diarization failed: %s", job_id, exc)

        segments = self._build_segments(whisper_result, merged_segments)
        sentiment = analyze_sentiment(transcript)

        if speaker_texts:
            sentiment["speaker_sentiments"] = analyze_per_speaker(speaker_texts)
        else:
            sentiment["speaker_sentiments"] = {}

        stats = self._compute_statistics(transcript, segments)

        pipeline_cfg = self.cfg.get("pipeline", {})
        output_dir = Path(
            resolve_path(
                pipeline_cfg.get(
                    "analysis_output_folder", self.cfg.get("output_folder", "./Output")
                )
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_analysis.json"

        payload = {
            "job_id": job_id,
            "status": "completed",
            "model": self.model_name,
            "source_audio": {
                "original_filename": original_filename,
                "stored_path": str(file_path),
            },
            "statistics": stats,
            "overall_sentiment": sentiment.get("overall", {}),
            "tones": sentiment.get("tones", []),
            "critical_moments": sentiment.get("critical_moments", []),
            "speaker_sentiments": sentiment.get("speaker_sentiments", {}),
            "segments": segments,
            "transcript": transcript,
        }

        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.chmod(output_path, 0o600)

        archived_path = self._archive_audio(job_id, file_path, original_filename)
        payload["source_audio"]["archived_path"] = archived_path
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        log.info("Job %s: wrote %s", job_id, output_path)
        return str(output_path)
