#!/usr/bin/env python3
"""
Acoustic Context Serializer for ClinicalWhisper
Converts numeric acoustic/prosodic features (eGeMAPSv02 via OpenSMILE)
into natural-language clinical narratives suitable for injection into
LLM prompts.  All processing is deterministic — no models loaded.

Feature dictionary keys match the output of AcousticExtractor._extract_metrics():
  pitch_mean_st, pitch_cv, loudness_mean_db, loudness_cv,
  jitter, shimmer, vta
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("ClinicalWhisper.acoustic_context")


# ---------------------------------------------------------------------------
# Clinical reference ranges
# ---------------------------------------------------------------------------
# Each entry: (low_threshold, high_threshold, low_label, normal_label, high_label)
# These are approximate ranges for spontaneous conversational speech.

_RANGES: dict[str, tuple[float, float, str, str, str]] = {
    # Conversational F0 in semitones re 27.5 Hz.
    # Adult males ≈ 7-12 st, females ≈ 12-20 st.  8-16 st captures the
    # central range for mixed-gender clinical recordings.
    "pitch_mean_st": (8.0, 16.0, "low", "normal", "high"),

    # Coefficient of variation of F0.  CV < 0.10 indicates monotone /
    # flat speech, common in depression and Parkinson's.  CV > 0.30
    # indicates unusually variable pitch (e.g., mania, distress).
    "pitch_cv": (0.10, 0.30, "low", "normal", "high"),

    # CV of loudness (energy).  Same interpretive frame as pitch_cv.
    # Low CV → monotone intensity, high CV → erratic volume control.
    "loudness_cv": (0.10, 0.30, "low", "normal", "high"),

    # Local jitter (cycle-to-cycle F0 perturbation).  Healthy speakers
    # typically <1 %.  1-3 % is mildly elevated (possible vocal strain,
    # fatigue, or early pathology).  >3 % is clinically elevated.
    "jitter": (0.01, 0.03, "normal", "mildly elevated", "elevated"),

    # Local shimmer (cycle-to-cycle amplitude perturbation, dB scale).
    # <3 % normal, 3-6 % mild, >6 % elevated — parallels jitter logic.
    "shimmer": (0.03, 0.06, "normal", "mildly elevated", "elevated"),

    # VTA (Vocal Tract Anhedonia index).  V_anh = -log(CV_F0 × CV_Energy).
    # Lower values → less prosodic variability → possible anhedonia.
    # <1.5 is concerning (flat affect), 1.5-3.0 is typical, >3.0 is
    # unusually expressive (could be mania or high engagement).
    "vta": (1.5, 3.0, "low", "normal", "high"),
}

# Human-readable clinical gloss for extreme categories.
_CLINICAL_GLOSS: dict[str, dict[str, str]] = {
    "pitch_mean_st": {
        "low": "below typical conversational range",
        "high": "above typical conversational range",
    },
    "pitch_cv": {
        "low": "suggests reduced prosodic variation, consistent with flat affect",
        "high": "unusually variable pitch — possible emotional lability or distress",
    },
    "loudness_cv": {
        "low": "monotone intensity — reduced dynamic range",
        "high": "erratic volume control — possible agitation or dysarthria",
    },
    "jitter": {
        "mildly elevated": "slight pitch perturbation — possible vocal fatigue",
        "elevated": "significant pitch instability — potential vocal pathology or distress",
    },
    "shimmer": {
        "mildly elevated": "slight amplitude perturbation — possible vocal fatigue",
        "elevated": "significant amplitude instability — potential vocal pathology",
    },
    "vta": {
        "low": "potential anhedonia indicator",
        "high": "elevated prosodic variability — high expressiveness or emotional lability",
    },
}

# Display names for human-readable output.
_DISPLAY_NAMES: dict[str, str] = {
    "pitch_mean_st": "Mean pitch",
    "pitch_cv": "Pitch variability (CV)",
    "loudness_mean_db": "Mean loudness",
    "loudness_cv": "Loudness variability (CV)",
    "jitter": "Jitter",
    "shimmer": "Shimmer",
    "vta": "VTA score",
}

# Units for numeric values.
_UNITS: dict[str, str] = {
    "pitch_mean_st": "st",
    "loudness_mean_db": "dB",
}


# ---------------------------------------------------------------------------
# 1. categorize_feature
# ---------------------------------------------------------------------------

def categorize_feature(feature_name: str, value: float) -> str:
    """Classify a single acoustic feature value as low / normal / high
    (or the range-specific labels like 'mildly elevated') based on
    clinical reference ranges for conversational speech.

    Args:
        feature_name: Key name matching AcousticExtractor output
                      (e.g. 'pitch_cv', 'jitter').
        value:        The numeric feature value.

    Returns:
        A category string (e.g. 'normal', 'low', 'elevated').
        Returns 'unknown' if the feature has no defined range.
    """
    if feature_name not in _RANGES:
        log.debug("No reference range defined for feature '%s'", feature_name)
        return "unknown"

    lo, hi, low_label, mid_label, high_label = _RANGES[feature_name]

    if value < lo:
        return low_label
    elif value > hi:
        return high_label
    return mid_label


# ---------------------------------------------------------------------------
# 2. serialize_overall_acoustics
# ---------------------------------------------------------------------------

def serialize_overall_acoustics(acoustics: dict) -> str:
    """Convert a full acoustic feature dictionary into a clinical narrative
    paragraph suitable for LLM prompt injection.

    Args:
        acoustics: Dict with keys matching AcousticExtractor output.
                   Missing keys are silently skipped.

    Returns:
        A multi-sentence clinical narrative string.  Returns a fallback
        message if the dict is empty or None.

    Example output:
        'Overall vocal profile: Mean pitch was 12.3 st (normal range).
        Pitch variability was 0.08 (low — suggests reduced prosodic
        variation, consistent with flat affect). ...'
    """
    if not acoustics:
        log.warning("serialize_overall_acoustics called with empty acoustics dict")
        return "Acoustic features were not available for this recording."

    parts: list[str] = []

    # --- Pitch block ---
    pitch_st = acoustics.get("pitch_mean_st")
    if pitch_st is not None:
        cat = categorize_feature("pitch_mean_st", pitch_st)
        unit = _UNITS.get("pitch_mean_st", "")
        desc = _format_category_phrase("pitch_mean_st", cat)
        parts.append(f"Mean pitch was {pitch_st} {unit} ({desc})")

    pitch_cv = acoustics.get("pitch_cv")
    if pitch_cv is not None:
        cat = categorize_feature("pitch_cv", pitch_cv)
        desc = _format_category_phrase("pitch_cv", cat)
        parts.append(f"Pitch variability was {pitch_cv} ({desc})")

    # --- Loudness block ---
    loudness_cv = acoustics.get("loudness_cv")
    if loudness_cv is not None:
        cat = categorize_feature("loudness_cv", loudness_cv)
        desc = _format_category_phrase("loudness_cv", cat)
        parts.append(f"Loudness variability was {loudness_cv} ({desc})")

    # --- Voice quality (jitter + shimmer) ---
    jitter = acoustics.get("jitter")
    shimmer = acoustics.get("shimmer")
    vq_parts: list[str] = []
    if jitter is not None:
        cat = categorize_feature("jitter", jitter)
        vq_parts.append(f"jitter {jitter} ({cat})")
    if shimmer is not None:
        cat = categorize_feature("shimmer", shimmer)
        vq_parts.append(f"shimmer {shimmer} ({cat})")
    if vq_parts:
        parts.append("Voice quality: " + ", ".join(vq_parts))

    # --- VTA ---
    vta = acoustics.get("vta")
    if vta is not None:
        cat = categorize_feature("vta", vta)
        desc = _format_category_phrase("vta", cat)
        parts.append(f"VTA score: {vta} ({desc})")

    if not parts:
        return "Acoustic features were not available for this recording."

    narrative = "Overall vocal profile: " + ". ".join(parts) + "."
    log.debug("Serialized overall acoustics (%d features)", len(parts))
    return narrative


# ---------------------------------------------------------------------------
# 3. serialize_segment_acoustics
# ---------------------------------------------------------------------------

def serialize_segment_acoustics(segment_acoustics: dict) -> str:
    """Produce a compact one-line summary for a single segment's acoustic
    features (e.g., one diarized speaker turn).

    Args:
        segment_acoustics: Dict with keys matching AcousticExtractor output.

    Returns:
        A concise single-line string.

    Example output:
        'Pitch: 10.2 st (normal), Loudness: normal variability, Voice quality: stable'
    """
    if not segment_acoustics:
        return "No acoustic data for this segment."

    tokens: list[str] = []

    # Pitch
    pitch_st = segment_acoustics.get("pitch_mean_st")
    if pitch_st is not None:
        cat = categorize_feature("pitch_mean_st", pitch_st)
        unit = _UNITS.get("pitch_mean_st", "")
        tokens.append(f"Pitch: {pitch_st} {unit} ({cat})")

    # Loudness variability
    loudness_cv = segment_acoustics.get("loudness_cv")
    if loudness_cv is not None:
        cat = categorize_feature("loudness_cv", loudness_cv)
        tokens.append(f"Loudness: {cat} variability")

    # Voice quality — summarize jitter + shimmer into a single word
    jitter = segment_acoustics.get("jitter")
    shimmer = segment_acoustics.get("shimmer")
    vq_label = _voice_quality_summary(jitter, shimmer)
    tokens.append(f"Voice quality: {vq_label}")

    summary = ", ".join(tokens)
    log.debug("Serialized segment acoustics: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# 4. build_acoustic_prompt_context
# ---------------------------------------------------------------------------

def build_acoustic_prompt_context(
    overall: dict,
    speaker_acoustics: dict[str, dict],
) -> str:
    """Combine overall and per-speaker acoustics into a single markdown
    block ready for injection into an LLM clinical scoring prompt.

    Args:
        overall:           Full-recording acoustic feature dict.
        speaker_acoustics: Mapping of speaker label (e.g. 'SPEAKER_00')
                           to that speaker's acoustic feature dict.

    Returns:
        A markdown-formatted string with ## / ### headers.

    Example structure:
        ## Acoustic Analysis
        ### Overall
        Overall vocal profile: ...
        ### By Speaker
        **SPEAKER_00**: Pitch: 10.2 st (normal), ...
        **SPEAKER_01**: Pitch: 14.5 st (normal), ...
    """
    sections: list[str] = ["## Acoustic Analysis"]

    # Overall block
    sections.append("### Overall")
    sections.append(serialize_overall_acoustics(overall))

    # Per-speaker block
    if speaker_acoustics:
        sections.append("### By Speaker")
        for speaker, feats in sorted(speaker_acoustics.items()):
            line = serialize_segment_acoustics(feats)
            sections.append(f"**{speaker}**: {line}")
    else:
        log.debug("No per-speaker acoustics provided; skipping speaker section")

    prompt_block = "\n".join(sections)
    log.info(
        "Built acoustic prompt context: %d chars, %d speakers",
        len(prompt_block),
        len(speaker_acoustics) if speaker_acoustics else 0,
    )
    return prompt_block


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_category_phrase(feature_name: str, category: str) -> str:
    """Build a human-readable phrase for a feature's category,
    optionally appending the clinical gloss.

    For 'normal'-equivalent categories, just returns the range label.
    For abnormal categories, appends the gloss separated by ' — '.
    """
    gloss_map = _CLINICAL_GLOSS.get(feature_name, {})
    gloss = gloss_map.get(category)

    # "normal" / "normal range" style
    if category in ("normal",):
        return "normal range"
    if gloss:
        return f"{category} — {gloss}"
    return category


def _voice_quality_summary(
    jitter: Optional[float],
    shimmer: Optional[float],
) -> str:
    """Collapse jitter + shimmer into a single voice-quality word.

    Returns one of: 'stable', 'mildly unstable', 'unstable', or
    'indeterminate' if neither metric is available.
    """
    if jitter is None and shimmer is None:
        return "indeterminate"

    severity = 0  # 0 = normal, 1 = mild, 2 = elevated

    if jitter is not None:
        jcat = categorize_feature("jitter", jitter)
        if jcat == "elevated":
            severity = max(severity, 2)
        elif jcat == "mildly elevated":
            severity = max(severity, 1)

    if shimmer is not None:
        scat = categorize_feature("shimmer", shimmer)
        if scat == "elevated":
            severity = max(severity, 2)
        elif scat == "mildly elevated":
            severity = max(severity, 1)

    return {0: "stable", 1: "mildly unstable", 2: "unstable"}[severity]


# ---------------------------------------------------------------------------
# CLI test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Synthetic test data that exercises all code paths.
    sample_overall = {
        "pitch_mean_st": 12.3,
        "pitch_cv": 0.08,
        "loudness_mean_db": -22.5,
        "loudness_cv": 0.22,
        "jitter": 0.015,
        "shimmer": 0.042,
        "vta": 1.2,
    }

    sample_speakers = {
        "SPEAKER_00": {
            "pitch_mean_st": 10.2,
            "pitch_cv": 0.18,
            "loudness_cv": 0.25,
            "jitter": 0.008,
            "shimmer": 0.022,
            "vta": 2.1,
        },
        "SPEAKER_01": {
            "pitch_mean_st": 14.5,
            "pitch_cv": 0.35,
            "loudness_cv": 0.09,
            "jitter": 0.045,
            "shimmer": 0.072,
            "vta": 0.9,
        },
    }

    print("=" * 72)
    print("TEST 1 — categorize_feature")
    print("=" * 72)
    for feat, val in sample_overall.items():
        cat = categorize_feature(feat, val)
        print(f"  {feat:>18} = {val:>8.3f}  →  {cat}")

    print()
    print("=" * 72)
    print("TEST 2 — serialize_overall_acoustics")
    print("=" * 72)
    print(serialize_overall_acoustics(sample_overall))

    print()
    print("=" * 72)
    print("TEST 3 — serialize_segment_acoustics")
    print("=" * 72)
    for spk, feats in sample_speakers.items():
        print(f"  {spk}: {serialize_segment_acoustics(feats)}")

    print()
    print("=" * 72)
    print("TEST 4 — build_acoustic_prompt_context")
    print("=" * 72)
    print(build_acoustic_prompt_context(sample_overall, sample_speakers))

    print()
    print("=" * 72)
    print("TEST 5 — edge cases")
    print("=" * 72)
    print("  Empty dict:", serialize_overall_acoustics({}))
    print("  None dict: ", serialize_overall_acoustics(None))
    print("  Unknown feat:", categorize_feature("nonexistent", 42.0))
    print("  Empty segment:", serialize_segment_acoustics({}))
