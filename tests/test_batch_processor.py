"""Tests for summary CSV extraction."""

import json

from batch_processor import _extract_row


def test_extract_row_surfaces_llm_failure(tmp_path):
    analysis = {
        "status": "completed_with_warnings",
        "warnings": ["LLM clinical scoring failed: model unavailable"],
        "statistics": {"word_count": 42, "duration_seconds": 60},
        "llm_scoring_status": "failed",
        "llm_scoring_coverage": 0.0,
        "llm_clinical_scoring": {
            "anhedonia_content": None,
            "depressed_mood_content": None,
            "affect_flatness": None,
            "engagement_level": None,
            "_meta": {"error": "model unavailable"},
        },
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis), encoding="utf-8")

    row = _extract_row(str(path), "interview.mp3")

    assert row["status"] == "completed_with_warnings"
    assert row["llm_scoring_status"] == "failed"
    assert row["llm_scoring_coverage"] == 0.0
    assert "model unavailable" in row["warnings"]


def test_extract_row_classifies_legacy_partial_result(tmp_path):
    analysis = {
        "status": "completed",
        "llm_clinical_scoring": {
            "anhedonia_content": 2,
            "_meta": {"runs_failed": 1, "coverage": 0.5},
        },
    }
    path = tmp_path / "legacy_analysis.json"
    path.write_text(json.dumps(analysis), encoding="utf-8")

    row = _extract_row(str(path), "interview.mp3")

    assert row["llm_scoring_status"] == "partial"
    assert row["llm_scoring_coverage"] == 0.5
