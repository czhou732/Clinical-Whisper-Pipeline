"""The scrubber must survive segments OpenMED's input guard would reject."""

from types import SimpleNamespace

import pii_scrubber
from pii_scrubber import PIIScrubber


def _scrubber(monkeypatch):
    calls = []

    def fake_deidentify(text, **kwargs):
        calls.append(text)
        return SimpleNamespace(deidentified_text=text.replace("John", "[NAME]"), pii_entities=[1])

    monkeypatch.setattr(pii_scrubber, "deidentify", fake_deidentify)
    s = object.__new__(PIIScrubber)
    s.is_available, s.strict, s.model_name, s.confidence_threshold = True, True, "m", 0.7
    s._config, s.entity_count, s.redacted_count, s.lang = None, 0, 0, "en"
    s.entity_types, s._ids = {}, {}
    return s, calls


def test_punctuation_only_segments_pass_through_without_calling_openmed(monkeypatch):
    s, calls = _scrubber(monkeypatch)
    assert s.scrub_text("...") == "..."
    assert s.scrub_text(" ? ") == " ? "
    assert calls == []


def test_mostly_punctuation_with_letters_is_redacted_not_leaked(monkeypatch):
    s, calls = _scrubber(monkeypatch)
    assert s.scrub_text("J...?!") == "[REDACTED]"
    assert calls == [] and s.redacted_count == 1


def test_repeated_character_garbage_is_collapsed_then_scrubbed(monkeypatch):
    s, calls = _scrubber(monkeypatch)
    out = s.scrub_text("John said " + "a" * 150)
    assert out == "[NAME] said aaa" and len(calls) == 1


def test_normal_text_is_scrubbed(monkeypatch):
    s, calls = _scrubber(monkeypatch)
    assert s.scrub_text("My name is John.") == "My name is [NAME]."
