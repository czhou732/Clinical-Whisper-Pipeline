"""The provenance block must record the masking settings a run actually used."""

import inspect

import inference_pipeline
import provenance
from pii_scrubber import PIIScrubber


def test_default_threshold_matches_the_pipeline(monkeypatch):
    monkeypatch.setattr(provenance, "_hf_revision", lambda repo: None)
    record = provenance.build({})
    # The pipeline masks at 0.5 when the config gives no threshold; provenance
    # used to report 0.7 for those runs.
    assert "pii_cfg.get(\"confidence_threshold\", 0.5)" in inspect.getsource(inference_pipeline)
    assert record["models"]["deidentification"]["confidence_threshold"] == 0.5


def test_summary_names_the_masker_and_threshold_used():
    s = object.__new__(PIIScrubber)
    s.model_name, s.confidence_threshold = "OpenMed/some-model", 0.3
    s.entity_count, s.entity_types, s._ids = 0, {}, {}
    s.redacted_count, s.second_check_added = 0, None
    summary = s.summary()
    assert summary["model"] == "OpenMed/some-model"
    assert summary["confidence_threshold"] == 0.3
