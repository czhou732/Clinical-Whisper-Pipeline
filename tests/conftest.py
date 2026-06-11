"""Shared fixtures for ClinicalWhisper tests."""

import pytest
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def positive_text():
    return "I'm feeling great today! Everything is going wonderfully."


@pytest.fixture
def negative_text():
    return "I can't seem to enjoy anything anymore. Everything feels empty and meaningless."


@pytest.fixture
def clinical_masking_text():
    return "I'm doing fine. Everything is okay I guess."


@pytest.fixture
def neutral_text():
    return "The weather forecast says it will be sunny tomorrow."


@pytest.fixture
def sample_segments():
    return [
        {"start": 0.0, "end": 5.2, "speaker": "Speaker 1", "text": "Hello, how are you today?"},
        {"start": 5.3, "end": 12.1, "speaker": "Speaker 2", "text": "I'm doing okay, thanks for asking."},
        {"start": 12.2, "end": 20.5, "speaker": "Speaker 1", "text": "Tell me about your week."},
        {"start": 20.6, "end": 35.0, "speaker": "Speaker 2", "text": "It was really tough. I couldn't sleep at all."},
    ]


@pytest.fixture
def sample_transcript():
    return (
        "Hello, how are you today? I'm doing okay, thanks for asking. "
        "Tell me about your week. It was really tough. I couldn't sleep at all. "
        "I've been feeling exhausted and can't seem to find motivation. "
        "Let's work together on some strategies. That sounds helpful, thank you. "
        "Great, I'm glad you're open to trying. We'll take this one step at a time."
    )
