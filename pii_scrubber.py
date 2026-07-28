#!/usr/bin/env python3
"""
PII Scrubber Module for ClinicalWhisper.

Uses OpenMED's local de-identification model to mask HIPAA Safe Harbor
identifiers (names, dates, addresses, phone numbers, MRNs, ...) in transcripts.

Runs entirely on-device. The detection model
(``OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1``, ~44M params) is public and
downloaded to ``~/.cache/huggingface`` on first use.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from openmed import OpenMedConfig, deidentify
except ImportError:  # pragma: no cover - exercised only when openmed is missing
    OpenMedConfig = None
    deidentify = None

try:
    import bundled_models
except ImportError:  # pragma: no cover
    bundled_models = None

log = logging.getLogger("ClinicalWhisper")

DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


class PIIScrubber:
    """Masks PII in transcript text using OpenMED.

    Failures are surfaced rather than silently returning the original text —
    a scrubber that quietly passes PHI through is worse than one that stops.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PII_MODEL,
        confidence_threshold: float = 0.7,
        strict: bool = True,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.strict = strict
        self.is_available = deidentify is not None
        self.entity_count = 0

        # When the app ships its own weights, point OpenMED at them so it never
        # looks in ~/.cache or reaches for the network.
        self._config = None
        cache_dir = bundled_models.openmed_cache_dir() if bundled_models else None
        if cache_dir and OpenMedConfig is not None:
            self._config = OpenMedConfig(cache_dir=cache_dir, local_only=True)
            log.info("Using bundled OpenMED weights.")

        if not self.is_available:
            log.warning("openmed package not found. PII scrubbing is disabled.")

    def scrub_text(self, text: str) -> str:
        """Mask PII in a single string.

        Raises:
            RuntimeError: when ``strict`` and de-identification fails, so the
            caller never mistakes unscrubbed text for scrubbed text.
        """
        if not self.is_available or not text or not text.strip():
            return text

        try:
            result = deidentify(
                text,
                method="mask",
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                config=self._config,
            )
            self.entity_count += len(getattr(result, "pii_entities", []) or [])
            return result.deidentified_text
        except Exception as e:
            log.error("Error during PII scrubbing: %s", e)
            if self.strict:
                raise RuntimeError(f"PII scrubbing failed: {e}") from e
            return text

    def scrub_segments(self, segments: list[dict]) -> list[dict]:
        """Mask PII across a list of MOSS segments.

        Expected format: ``[{"speaker": ..., "text": ..., "start": ..., "end": ...}]``
        """
        if not self.is_available or not segments:
            return segments

        log.info("Scrubbing PII from %d transcript segments...", len(segments))
        self.entity_count = 0

        scrubbed = []
        for seg in segments:
            new_seg = dict(seg)
            new_seg["text"] = self.scrub_text(seg.get("text", ""))
            scrubbed.append(new_seg)

        log.info("PII scrubbing complete — %d identifiers masked.", self.entity_count)
        return scrubbed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_text = (
        "My name is John Doe and my phone number is 555-231-9987. "
        "I live in New York and my MRN is 44821903."
    )
    scrubber = PIIScrubber()
    print("Original:", test_text)
    print("Scrubbed:", scrubber.scrub_text(test_text))
