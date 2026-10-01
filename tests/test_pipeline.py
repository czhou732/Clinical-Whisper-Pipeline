"""Tests for inference_pipeline._compute_statistics and related helpers."""

import json
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_pipeline import InferencePipeline


class TestComputeStatistics:
    """Tests for _compute_statistics static method."""

    def test_basic_statistics(self):
        transcript = "Hello world. This is a test sentence. Another one here."
        segments = [
            {"start": 0.0, "end": 10.5, "speaker": "Speaker 1", "text": "Hello world."},
            {"start": 10.6, "end": 25.0, "speaker": "Speaker 1", "text": "This is a test sentence."},
        ]
        stats = InferencePipeline._compute_statistics(transcript, segments)

        assert stats["word_count"] == 10
        assert stats["character_count"] == len(transcript)
        assert stats["sentence_count"] >= 1
        assert stats["duration_seconds"] == 25.0
        assert "estimated_minutes" in stats

    def test_sentence_counting_abbreviations(self):
        """NLTK should NOT split Dr. or D.C. into separate sentences."""
        transcript = "Dr. Smith went to Washington D.C. yesterday."
        segments = []
        stats = InferencePipeline._compute_statistics(transcript, segments)
        assert stats["sentence_count"] == 1

    def test_duration_from_segments(self):
        """Duration should come from segment timestamps, not word count estimation."""
        transcript = "Short text."
        segments = [
            {"start": 0.0, "end": 120.0, "speaker": "Speaker 1", "text": "Short text."},
        ]
        stats = InferencePipeline._compute_statistics(transcript, segments)
        assert stats["duration_seconds"] == 120.0
        assert stats["estimated_minutes"] == 2.0

    def test_empty_transcript(self):
        transcript = ""
        segments = []
        stats = InferencePipeline._compute_statistics(transcript, segments)
        assert stats["word_count"] == 0
        assert stats["sentence_count"] >= 0
        assert stats["duration_seconds"] == 0.0
        assert stats["estimated_minutes"] == 0.0

    def test_statistics_keys(self):
        transcript = "A simple test."
        segments = [{"start": 0.0, "end": 5.0, "speaker": "Speaker 1", "text": "A simple test."}]
        stats = InferencePipeline._compute_statistics(transcript, segments)
        expected_keys = {"word_count", "character_count", "sentence_count", "estimated_minutes", "duration_seconds"}
        assert set(stats.keys()) == expected_keys

    def test_no_segments_uses_word_count_fallback(self):
        """When there are no segments (duration_seconds=0), fall back to word count / 150."""
        transcript = "This is a sentence with exactly nine words here."
        segments = []
        stats = InferencePipeline._compute_statistics(transcript, segments)
        assert stats["duration_seconds"] == 0.0
        assert stats["estimated_minutes"] == round(9 / 150.0, 2)


class TestAssessLLMScoring:
    """Scorer failures returned in metadata must not look like clean results."""

    def test_total_failure_is_reported(self):
        result = {
            "hesitancy_score": 0,
            "_meta": {"error": "model unavailable", "windows": 2},
        }

        status, coverage, warnings = InferencePipeline._assess_llm_scoring(result)

        assert status == "failed"
        assert coverage == 0.0
        assert warnings == ["LLM clinical scoring failed: model unavailable"]

    def test_partial_failure_reports_coverage(self):
        result = {
            "hesitancy_score": 4,
            "_meta": {"errors": ["window 2 failed"], "coverage": 0.5},
        }

        status, coverage, warnings = InferencePipeline._assess_llm_scoring(result)

        assert status == "partial"
        assert coverage == 0.5
        assert "1 scoring run(s) failed" in warnings[0]
        assert "50.0%" in warnings[0]

    def test_success_remains_clean(self):
        result = {"hesitancy_score": 4, "_meta": {"coverage": 1.0}}

        status, coverage, warnings = InferencePipeline._assess_llm_scoring(result)

        assert status == "completed"
        assert coverage == 1.0
        assert warnings == []

    def test_score_job_marks_returned_failure_as_warning(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "transcript_formatter",
            SimpleNamespace(
                process_segments=lambda segments: {
                    "structured_transcript": "[00:00] Subject: Test",
                    "roles": {"S01": "Subject"},
                    "speaker_stats": {},
                }
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "acoustic_context",
            SimpleNamespace(build_acoustic_prompt_context=lambda overall, speakers: ""),
        )
        monkeypatch.setitem(
            sys.modules,
            "llm_clinical_scorer",
            SimpleNamespace(
                score_transcript=lambda transcript, context, cfg: {
                    "hesitancy_score": 0,
                    "_meta": {"error": "model unavailable"},
                }
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "provenance",
            SimpleNamespace(build=lambda cfg: {"app_version": "test"}),
        )

        source = tmp_path / "interview.mp3"
        source.write_bytes(b"test")
        pipeline = InferencePipeline(
            {
                "pipeline": {"analysis_output_folder": str(tmp_path / "output")},
                "audio_retention": "delete",
                "llm_scoring": {"enabled": True},
            }
        )
        state = {
            "job": {"job_id": "test_job"},
            "job_id": "test_job",
            "file_path": source,
            "original_filename": source.name,
            "segments": [{"speaker": "S01", "text": "Test", "start": 0, "end": 1}],
            "transcript": "Test",
            "stats": {"word_count": 1, "duration_seconds": 1},
            "overall_acoustics": {},
            "speaker_acoustics": {},
            "warnings": [],
        }

        output_path = pipeline.score_job(state)
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

        assert payload["status"] == "completed_with_warnings"
        assert payload["llm_scoring_status"] == "failed"
        assert payload["llm_scoring_coverage"] == 0.0
        assert payload["warnings"] == [
            "LLM clinical scoring failed: model unavailable"
        ]
