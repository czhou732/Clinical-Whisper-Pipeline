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
import re
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

_ALNUM = re.compile(r"[^\W_]")
_TAG = re.compile(r"\[([a-z_]+)\]")
_UPPER_TAG = re.compile(r"\[([A-Z][A-Z_]*)\]")
# The single-language models label in capitals without separators; the
# English model uses snake_case. One vocabulary keeps the masking key, the
# tag numbering and name silencing working in every language.
_LABEL_NAMES = {"firstname": "first_name", "lastname": "last_name", "middlename": "middle_name",
                "zipcode": "zip_code", "phonenumber": "phone_number", "telephonenum": "phone_number",
                "streetaddress": "street_address", "street": "street_address",
                "buildingnum": "building_number", "dateofbirth": "date_of_birth",
                "idcard": "id_number", "socialnum": "ssn", "username": "username"}


def label_name(label: str) -> str:
    """A model's entity label in ClinicalWhisper's snake_case vocabulary."""
    low = str(label or "other").lower()
    return _LABEL_NAMES.get(low.replace("_", ""), low)
_REPEATS = re.compile(r"(.)\1{20,}")


def _mostly_punctuation(text: str) -> bool:
    """OpenMED's own rule: more than half the characters are neither word nor space."""
    return len(re.findall(r"[^\w\s]", text)) / len(text) > 0.5

DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


def number_tags(masked: str, entities: list, ids: dict[tuple[str, str], int]) -> str:
    """Turn ``[first_name]`` tags into ``[first_name_1]``, ``[first_name_2]``, ...

    The same identifier gets the same number throughout one scrubber's life
    (one recording), so mentions of a person can be counted and followed
    without the name. ``ids`` maps (type, identifier key) -> number and is
    updated in place; the key is OpenMED's HMAC of the normalised text, so the
    name itself is never stored.

    Tags are matched to entities in order. If they do not line up one to one —
    OpenMED merged spans, or a later sweep masked something the entity list
    does not show — the masked text is returned unnumbered: plain tags are
    safe, a wrong number is not.
    """
    tags = list(_TAG.finditer(masked))
    ents = sorted(entities, key=lambda e: getattr(e, "start", 0))
    if len(tags) != len(ents) or any(
        t.group(1) != label_name(getattr(e, "label", None)) for t, e in zip(tags, ents)
    ):
        return masked
    out, last = [], 0
    for tag, ent in zip(tags, ents):
        meta = getattr(ent, "metadata", None) or {}
        key = meta.get("normalized_text_hash") or str(getattr(ent, "text", "")).strip().lower()
        label = tag.group(1)
        n = ids.setdefault((label, key), 1 + sum(1 for lab, _ in ids if lab == label))
        out.append(masked[last:tag.start()])
        out.append(f"[{label}_{n}]")
        last = tag.end()
    out.append(masked[last:])
    return "".join(out)


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
        lang: str = "en",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        # Selects OpenMED's regex patterns (phone numbers, IDs) for the language.
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.strict = strict
        self.is_available = deidentify is not None
        self.entity_count = 0
        # Masked mentions per identifier type, e.g. {"first_name": 12}.
        self.entity_types: dict[str, int] = {}
        # (type, HMAC of identifier) -> number, for consistent numbered tags.
        self._ids: dict[tuple[str, str], int] = {}

        # When the app ships its own weights, point OpenMED at them so it never
        # looks in ~/.cache or reaches for the network.
        self._config = None
        self.redacted_count = 0
        cache_dir = cache_dir or (bundled_models.openmed_cache_dir() if bundled_models else None)
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

        # OpenMED's input guard rejects text that is mostly punctuation or has a
        # character repeated 100+ times. MOSS emits segments like "..." or "?"
        # on their own, and in strict mode one such segment failed the whole
        # file (about 1 recording in 10). Handle those cases before the call.
        if not _ALNUM.search(text):
            return text  # no letters or digits: nothing identifiable to mask
        text = _REPEATS.sub(lambda m: m.group(1) * 3, text)  # collapse decode garbage
        if _mostly_punctuation(text):
            # Some letters, but too few for the model to scrub: redact rather
            # than risk emitting an identifier unscrubbed.
            self.redacted_count += 1
            return "[REDACTED]"

        try:
            result = deidentify(
                text,
                method="mask",
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                config=self._config,
                lang=self.lang,
            )
            entities = list(getattr(result, "pii_entities", []) or [])
            self.entity_count += len(entities)
            for ent in entities:
                label = label_name(getattr(ent, "label", "other"))
                self.entity_types[label] = self.entity_types.get(label, 0) + 1
            masked = _UPPER_TAG.sub(lambda m: f"[{label_name(m.group(1))}]", result.deidentified_text)
            return number_tags(masked, entities, self._ids)
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
        self.redacted_count = 0
        self.entity_types = {}
        self._ids = {}

        scrubbed = []
        for seg in segments:
            new_seg = dict(seg)
            new_seg["text"] = self.scrub_text(seg.get("text", ""))
            scrubbed.append(new_seg)

        log.info("PII scrubbing complete — %d identifiers masked (%d distinct), "
                 "%d segment(s) redacted whole.",
                 self.entity_count, len(self._ids), self.redacted_count)
        return scrubbed

    def summary(self) -> dict:
        """Counts for the analysis output. Never includes the identifiers."""
        return {
            "masked_mentions": self.entity_count,
            "masked_by_type": dict(sorted(self.entity_types.items())),
            "distinct_identifiers": len(self._ids),
            "segments_redacted_whole": self.redacted_count,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_text = (
        "My name is John Doe and my phone number is 555-231-9987. "
        "I live in New York and my MRN is 44821903."
    )
    scrubber = PIIScrubber()
    print("Original:", test_text)
    print("Scrubbed:", scrubber.scrub_text(test_text))
