"""Tests for the sentiment_analyzer module (RoBERTa transformer backend)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiment_analyzer import (
    analyze_sentiment,
    analyze_per_speaker,
    format_sentiment_markdown,
    _score_bar,
    _get_segment_sentiment_score,
    _polarity_to_score,
    _label_from_scores,
    _detect_tones,
    _find_critical_moments,
)


# ---------------------------------------------------------------------------
# Basic sentiment detection
# ---------------------------------------------------------------------------

class TestSentimentDetection:
    def test_positive_detection(self, positive_text):
        result = analyze_sentiment(positive_text)
        assert result["overall"]["label"] == "Positive"
        assert result["overall"]["score"] >= 7

    def test_negative_detection(self, negative_text):
        result = analyze_sentiment(negative_text)
        assert result["overall"]["label"] == "Negative"
        assert result["overall"]["score"] <= 4

    def test_clinical_masking(self, clinical_masking_text):
        """Patient says 'fine' / 'okay I guess' — should NOT be classified as Positive."""
        result = analyze_sentiment(clinical_masking_text)
        assert result["overall"]["label"] != "Positive"

    def test_negated_positive(self):
        text = "The medication is working but I still feel empty inside."
        result = analyze_sentiment(text)
        # Should lean negative or neutral, not positive
        assert result["overall"]["label"] in ("Negative", "Neutral")
        assert result["overall"]["score"] <= 6

    def test_anhedonia_language(self):
        text = "I used to love hiking but now even the view feels meh."
        result = analyze_sentiment(text)
        assert result["overall"]["label"] in ("Negative", "Neutral")
        assert result["overall"]["score"] <= 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input(self):
        result = analyze_sentiment("")
        assert result["overall"]["label"] == "Neutral"
        assert result["overall"]["score"] == 5
        assert result["segments"] == []

    def test_none_input(self):
        result = analyze_sentiment(None)
        assert result["overall"]["label"] == "Neutral"
        assert result["overall"]["score"] == 5

    def test_whitespace_only(self):
        result = analyze_sentiment("   \n\t  ")
        assert result["overall"]["label"] == "Neutral"
        assert result["overall"]["score"] == 5


# ---------------------------------------------------------------------------
# Score range validation
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_positive_score_range(self, positive_text):
        result = analyze_sentiment(positive_text)
        assert 1 <= result["overall"]["score"] <= 10

    def test_negative_score_range(self, negative_text):
        result = analyze_sentiment(negative_text)
        assert 1 <= result["overall"]["score"] <= 10

    def test_neutral_score_range(self, neutral_text):
        result = analyze_sentiment(neutral_text)
        assert 1 <= result["overall"]["score"] <= 10

    def test_polarity_to_score_bounds(self):
        assert _polarity_to_score(-1.0) == 1
        assert _polarity_to_score(1.0) == 10
        assert _polarity_to_score(0.0) == 6  # rounds 5.5 up


# ---------------------------------------------------------------------------
# Segment and structure
# ---------------------------------------------------------------------------

class TestSegments:
    def test_segments_structure(self, sample_transcript):
        result = analyze_sentiment(sample_transcript)
        assert isinstance(result["segments"], list)
        for seg in result["segments"]:
            assert "index" in seg
            assert "score" in seg
            assert "label" in seg
            assert "polarity" in seg
            assert "preview" in seg
            assert 1 <= seg["score"] <= 10

    def test_return_structure(self, positive_text):
        result = analyze_sentiment(positive_text)
        assert "overall" in result
        assert "tones" in result
        assert "segments" in result
        assert "critical_moments" in result
        assert "speaker_sentiments" in result
        # Overall keys
        overall = result["overall"]
        assert "label" in overall
        assert "score" in overall
        assert "polarity" in overall
        assert "subjectivity" in overall


# ---------------------------------------------------------------------------
# Per-speaker
# ---------------------------------------------------------------------------

class TestPerSpeaker:
    def test_per_speaker(self):
        speaker_texts = {
            "Speaker 1": "Everything is wonderful and I feel great!",
            "Speaker 2": "I am so sad and everything is terrible.",
        }
        result = analyze_per_speaker(speaker_texts)
        assert "Speaker 1" in result
        assert "Speaker 2" in result
        for spk_data in result.values():
            assert "label" in spk_data
            assert "score" in spk_data
            assert "polarity" in spk_data
            assert "subjectivity" in spk_data
            assert "tones" in spk_data
            assert "word_count" in spk_data
            assert 1 <= spk_data["score"] <= 10

    def test_per_speaker_empty_text(self):
        speaker_texts = {"Speaker 1": "   "}
        result = analyze_per_speaker(speaker_texts)
        assert "Speaker 1" not in result


# ---------------------------------------------------------------------------
# Tone detection
# ---------------------------------------------------------------------------

class TestToneDetection:
    def test_enthusiastic_tone(self):
        text = "This is excited and amazing, so fantastic and incredible!"
        tones = _detect_tones(text)
        assert "enthusiastic" in tones

    def test_neutral_fallback(self):
        text = "The cat sat on the mat."
        tones = _detect_tones(text)
        assert tones == ["neutral"]


# ---------------------------------------------------------------------------
# Critical moments
# ---------------------------------------------------------------------------

class TestCriticalMoments:
    def test_critical_moments_detected(self):
        segments = [
            {"polarity": 0.8, "score": 9, "preview": "Great start"},
            {"polarity": -0.5, "score": 3, "preview": "Things went bad"},
            {"polarity": 0.6, "score": 8, "preview": "Recovery happened"},
        ]
        moments = _find_critical_moments(segments)
        assert len(moments) >= 1
        assert any(m["direction"] == "declined" for m in moments)
        assert any(m["direction"] == "improved" for m in moments)

    def test_no_critical_moments(self):
        segments = [
            {"polarity": 0.5, "score": 7, "preview": "Steady"},
            {"polarity": 0.55, "score": 8, "preview": "Still steady"},
        ]
        moments = _find_critical_moments(segments)
        assert len(moments) == 0


# ---------------------------------------------------------------------------
# Markdown and formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_markdown_output(self, sample_transcript):
        result = analyze_sentiment(sample_transcript)
        md = format_sentiment_markdown(result)
        assert isinstance(md, str)
        assert "Sentiment Analysis" in md
        assert "Overall Sentiment" in md

    def test_score_bar(self):
        bar = _score_bar(8)
        assert "8/10" in bar
        assert len(bar.split()[0]) == 10  # 8 filled + 2 empty = 10 chars

    def test_score_bar_boundaries(self):
        bar_min = _score_bar(1)
        assert "1/10" in bar_min
        bar_max = _score_bar(10)
        assert "10/10" in bar_max


# ---------------------------------------------------------------------------
# Segment sentiment score (used by inference_pipeline.py)
# ---------------------------------------------------------------------------

class TestSegmentSentimentScore:
    def test_returns_int(self):
        score = _get_segment_sentiment_score("I am very happy today!")
        assert isinstance(score, int)
        assert 1 <= score <= 10

    def test_empty_returns_default(self):
        assert _get_segment_sentiment_score("") == 5
        assert _get_segment_sentiment_score("   ") == 5

    def test_none_returns_default(self):
        assert _get_segment_sentiment_score(None) == 5
