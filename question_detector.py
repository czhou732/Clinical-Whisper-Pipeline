#!/usr/bin/env python3
"""
Question Detector — Clinical Interview Probe Categorization for ClinicalWhisper

Automatically identifies and categorizes interviewer questions/probes in
structured clinical interviews (e.g., DAIC-WOZ) using a local Ollama LLM.

Each interviewer turn is classified into a clinical probe category
(positive_affect, ptsd_screen, suicidal_ideation, etc.) and tagged segments
are returned so downstream modules can analyze responses per topic area.

Privacy-first: all classification runs locally via Ollama — no data leaves
the machine.

Usage:
  python question_detector.py --file /path/to/segments.json
  python question_detector.py --file /path/to/segments.json --model qwen2:7b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cw_config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("ClinicalWhisper")

# ---------------------------------------------------------------------------
# Clinical probe categories — standard for structured clinical interviews
# ---------------------------------------------------------------------------
CLINICAL_PROBE_CATEGORIES: dict[str, dict[str, Any]] = {
    "positive_affect": {
        "description": (
            "Questions about happiness, enjoyment, things going well, "
            "positive experiences, hobbies, or pleasurable activities."
        ),
        "examples": [
            "What makes you happy?",
            "Tell me about something you enjoy doing.",
            "What's been going well for you lately?",
        ],
        "critical": False,
    },
    "negative_affect": {
        "description": (
            "Questions about sadness, worry, stress, anger, frustration, "
            "hopelessness, or other negative emotional states."
        ),
        "examples": [
            "Do you feel sad or down?",
            "What worries you the most?",
            "How often do you feel stressed?",
        ],
        "critical": False,
    },
    "ptsd_screen": {
        "description": (
            "Questions about traumatic experiences, nightmares, flashbacks, "
            "hypervigilance, avoidance of triggers, or combat/deployment experiences."
        ),
        "examples": [
            "Have you experienced any traumatic events?",
            "Do you have nightmares?",
            "Are there places or situations you avoid because they remind you of something?",
        ],
        "critical": False,
    },
    "social_functioning": {
        "description": (
            "Questions about relationships, social activities, family, "
            "friends, loneliness, social support, or community involvement."
        ),
        "examples": [
            "How are your relationships with family?",
            "Do you spend time with friends?",
            "Do you feel lonely?",
        ],
        "critical": False,
    },
    "daily_functioning": {
        "description": (
            "Questions about work, school, daily routine, sleep quality, "
            "appetite, energy levels, or ability to carry out everyday tasks."
        ),
        "examples": [
            "How have you been sleeping?",
            "Tell me about your daily routine.",
            "Has your appetite changed?",
            "How is work going?",
        ],
        "critical": False,
    },
    "coping_strategies": {
        "description": (
            "Questions about how the person handles stress, self-care "
            "practices, substance use, exercise, therapy, or other "
            "mechanisms for managing difficulty."
        ),
        "examples": [
            "What do you do when you feel stressed?",
            "How do you take care of yourself?",
            "Do you exercise regularly?",
        ],
        "critical": False,
    },
    "suicidal_ideation": {
        "description": (
            "Questions about self-harm, suicidal thoughts, desire to die, "
            "plans for self-injury, or reasons for living. CRITICAL — "
            "always flag these segments for clinical review."
        ),
        "examples": [
            "Have you ever thought about hurting yourself?",
            "Do you have thoughts of ending your life?",
            "Have you made any plans to harm yourself?",
        ],
        "critical": True,
    },
    "rapport_building": {
        "description": (
            "Small talk, ice-breakers, non-clinical questions used to "
            "build trust and comfort. Greetings, weather, travel, or "
            "general introductory conversation."
        ),
        "examples": [
            "How are you doing today?",
            "Where are you from?",
            "Did you have a good weekend?",
        ],
        "critical": False,
    },
    "follow_up": {
        "description": (
            "Elaboration prompts that ask the participant to expand on "
            "a previous answer. Not a new topic — a continuation."
        ),
        "examples": [
            "Tell me more about that.",
            "Can you explain what you mean?",
            "How did that make you feel?",
            "What happened next?",
        ],
        "critical": False,
    },
    "other": {
        "description": (
            "Procedural statements, administrative instructions, or "
            "content that does not fit any clinical probe category."
        ),
        "examples": [
            "We're going to start the interview now.",
            "Let me check my notes.",
            "Thank you for your time today.",
        ],
        "critical": False,
    },
}

_VALID_CATEGORIES = set(CLINICAL_PROBE_CATEGORIES.keys())

# ---------------------------------------------------------------------------
# Ollama prompt template
# ---------------------------------------------------------------------------
QUESTION_DETECTION_PROMPT = """\
You are a clinical psychology research assistant analyzing a structured \
clinical interview transcript. Your task is to categorize each interviewer \
turn into exactly one clinical probe category.

## Probe Categories

{category_block}

## Interviewer Turns

Below are the interviewer's turns from the interview, each with an index \
number and the spoken text.

{turns_block}

## Instructions

1. For EACH interviewer turn above, determine which single probe category \
best matches the intent of the question or statement.
2. Assign a confidence score between 0.0 and 1.0 for your classification.
3. If a turn contains multiple topics, choose the PRIMARY intent.
4. "follow_up" should ONLY be used for generic elaboration prompts \
("tell me more", "go on", "can you explain") that do not introduce a \
new topic. If a follow-up question introduces a specific clinical topic, \
classify it under that topic instead.
5. Non-question statements by the interviewer (greetings, transitions, \
acknowledgments like "okay", "I see", "thank you") should be classified \
as "rapport_building" or "other" as appropriate.

## Output Format

Return ONLY a valid JSON array with no surrounding text, markdown, or \
explanation. Each element must have exactly these keys:
- "turn_index": integer (matching the index numbers above)
- "text": string (the original text of the turn)
- "category": string (one of the category names listed above)
- "confidence": float (0.0 to 1.0)

Example output format:
[
  {{"turn_index": 0, "text": "How are you today?", "category": "rapport_building", "confidence": 0.95}},
  {{"turn_index": 1, "text": "Do you feel sad?", "category": "negative_affect", "confidence": 0.90}}
]
"""


def _build_category_block() -> str:
    """Format probe categories for the LLM prompt."""
    lines: list[str] = []
    for name, info in CLINICAL_PROBE_CATEGORIES.items():
        critical_tag = " [CRITICAL]" if info["critical"] else ""
        examples = "; ".join(f'"{e}"' for e in info["examples"])
        lines.append(
            f"- **{name}**{critical_tag}: {info['description']} "
            f"Examples: {examples}"
        )
    return "\n".join(lines)


def _build_turns_block(turns: list[dict[str, Any]]) -> str:
    """Format interviewer turns as a numbered list for the prompt."""
    lines: list[str] = []
    for i, turn in enumerate(turns):
        text = turn.get("text", "").strip()
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama integration (mirrors meeting_intel.py pattern)
# ---------------------------------------------------------------------------
OLLAMA_GENERATE_ENDPOINT = "/api/generate"


def _call_ollama(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int = 180,
    max_retries: int = 1,
) -> str:
    """
    POST a prompt to the local Ollama /api/generate endpoint.

    Raises
    ------
    requests.exceptions.ConnectionError
        If Ollama is not running.
    RuntimeError
        If Ollama returns done=False, an empty response, or all retries fail.
    """
    url = base_url.rstrip("/") + OLLAMA_GENERATE_ENDPOINT
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": 4096,
        },
    }

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            log.info(
                "   Retrying Ollama call (attempt %d/%d)...",
                attempt + 1,
                max_retries + 1,
            )
            time.sleep(3)

        try:
            log.info(
                "   Calling Ollama for probe detection (model=%s, timeout=%ds)...",
                model,
                timeout,
            )
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            elapsed = time.time() - t0
            resp.raise_for_status()

            data = resp.json()
            if not data.get("done", False):
                raise RuntimeError(
                    "Ollama returned done=False — generation incomplete."
                )

            log.info("   Ollama probe detection response received (%.1fs)", elapsed)
            content = data.get("response", "")
            if not content.strip():
                raise RuntimeError("Ollama returned an empty response.")
            return content

        except requests.exceptions.ConnectionError:
            log.error(
                "   Ollama is not running at %s. Start it with: ollama serve",
                base_url,
            )
            raise

        except (requests.exceptions.Timeout, RuntimeError) as exc:
            log.warning(
                "   Ollama call failed (attempt %d): %s", attempt + 1, exc
            )
            last_exc = exc

    raise RuntimeError(
        f"Ollama probe detection failed after {max_retries + 1} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _parse_probe_json(raw_response: str, expected_count: int) -> list[dict[str, Any]]:
    """
    Extract and validate the JSON array from an Ollama response.

    Applies lenient parsing: tries the whole response first, then looks for
    a bracketed JSON array substring. Validates category names and confidence
    ranges, applying safe defaults for malformed entries.
    """
    # Try parsing the full response first
    text = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (possibly with language tag)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed: Optional[list[dict[str, Any]]] = None

    # Attempt 1: parse the whole text as JSON
    try:
        candidate = json.loads(text)
        if isinstance(candidate, list):
            parsed = candidate
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first [...] substring
    if parsed is None:
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            json_str = text[bracket_start : bracket_end + 1]
            try:
                candidate = json.loads(json_str)
                if isinstance(candidate, list):
                    parsed = candidate
            except json.JSONDecodeError:
                pass

    # Attempt 3: fix common issues — trailing commas before ]
    if parsed is None and bracket_start != -1 and bracket_end > bracket_start:
        json_str = text[bracket_start : bracket_end + 1]
        cleaned = re.sub(r",\s*]", "]", json_str)
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            candidate = json.loads(cleaned)
            if isinstance(candidate, list):
                parsed = candidate
        except json.JSONDecodeError:
            pass

    if parsed is None:
        raise ValueError(
            f"Could not extract JSON array from Ollama response. "
            f"Preview: {raw_response[:300]!r}"
        )

    # Validate and normalize each entry
    validated: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            log.warning("   Skipping non-dict entry in probe response: %r", item)
            continue

        turn_index = item.get("turn_index")
        if turn_index is None:
            log.warning("   Skipping entry with missing turn_index: %r", item)
            continue

        try:
            turn_index = int(turn_index)
        except (TypeError, ValueError):
            log.warning("   Skipping entry with invalid turn_index: %r", turn_index)
            continue

        category = str(item.get("category", "other")).lower().strip()
        if category not in _VALID_CATEGORIES:
            log.warning(
                "   Unknown category '%s' for turn %d — falling back to 'other'",
                category,
                turn_index,
            )
            category = "other"

        confidence = item.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        validated.append({
            "turn_index": turn_index,
            "text": str(item.get("text", "")),
            "category": category,
            "confidence": round(confidence, 3),
        })

    if not validated:
        raise ValueError(
            "Ollama returned a JSON array but no valid probe entries were found."
        )

    if len(validated) < expected_count:
        log.warning(
            "   LLM returned %d probe tags but expected %d — some turns may be untagged",
            len(validated),
            expected_count,
        )

    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_probes(
    segments: list[dict[str, Any]],
    roles: dict[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Detect and categorize clinical probes from interviewer turns.

    Parameters
    ----------
    segments : list[dict]
        Transcript segments, each with at least ``speaker``, ``start``,
        ``end``, and ``text`` keys (as produced by speaker_diarizer).
    roles : dict[str, str]
        Mapping of speaker labels to roles. Expected values include
        ``"interviewer"`` and ``"subject"`` (case-insensitive).
        Example: ``{"Speaker 1": "interviewer", "Speaker 2": "subject"}``.
    config : dict
        Full ClinicalWhisper config dict. Uses the ``question_detection``
        section for Ollama model/URL settings.

    Returns
    -------
    list[dict]
        Tagged interviewer turns, each containing:
        ``start``, ``end``, ``text``, ``category``, ``confidence``,
        ``is_critical``, ``speaker``.

    Notes
    -----
    Returns an empty list (without crashing) if there are no interviewer
    turns, if Ollama is unreachable, or if the LLM returns unparseable JSON.
    """
    qd_cfg = config.get("question_detection", {})
    if not qd_cfg.get("enabled", True):
        log.info("Question detection is disabled in config.")
        return []

    model = qd_cfg.get("ollama_model", "qwen2:7b")
    base_url = qd_cfg.get("ollama_base_url", "http://localhost:11434")
    timeout = qd_cfg.get("timeout_seconds", 180)
    max_retries = qd_cfg.get("max_retries", 1)

    # Identify interviewer speakers
    interviewer_speakers: set[str] = set()
    roles_lower = {spk: role.lower().strip() for spk, role in roles.items()}
    for spk, role in roles_lower.items():
        if role in ("interviewer", "clinician", "therapist", "ellie"):
            interviewer_speakers.add(spk)

    if not interviewer_speakers:
        log.warning(
            "No interviewer role found in roles mapping %s — "
            "cannot detect probes. Assign roles with "
            "{'Speaker X': 'interviewer'} before calling detect_probes.",
            roles,
        )
        return []

    # Filter to interviewer turns
    interviewer_turns: list[dict[str, Any]] = []
    for seg in segments:
        if seg.get("speaker") in interviewer_speakers:
            text = seg.get("text", "").strip()
            if text:
                interviewer_turns.append(seg)

    if not interviewer_turns:
        log.warning("No interviewer turns found in %d segments.", len(segments))
        return []

    log.info(
        "Detecting probes in %d interviewer turns (speakers: %s)...",
        len(interviewer_turns),
        ", ".join(sorted(interviewer_speakers)),
    )

    # Build prompt
    category_block = _build_category_block()
    turns_block = _build_turns_block(interviewer_turns)
    prompt = QUESTION_DETECTION_PROMPT.format(
        category_block=category_block,
        turns_block=turns_block,
    )

    # Call Ollama
    try:
        raw_response = _call_ollama(
            prompt=prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama for probe detection. Is it running?")
        return []
    except RuntimeError as exc:
        log.error("Ollama probe detection failed: %s", exc)
        return []

    # Parse response
    try:
        probe_tags = _parse_probe_json(raw_response, expected_count=len(interviewer_turns))
    except ValueError as exc:
        log.error("Failed to parse probe detection response: %s", exc)
        return []

    # Merge LLM tags back onto the original segment data
    result: list[dict[str, Any]] = []
    for tag in probe_tags:
        idx = tag["turn_index"]
        if idx < 0 or idx >= len(interviewer_turns):
            log.warning("   turn_index %d out of range (0-%d), skipping", idx, len(interviewer_turns) - 1)
            continue

        source_seg = interviewer_turns[idx]
        is_critical = CLINICAL_PROBE_CATEGORIES.get(tag["category"], {}).get("critical", False)

        result.append({
            "start": source_seg["start"],
            "end": source_seg["end"],
            "text": source_seg.get("text", "").strip(),
            "speaker": source_seg.get("speaker", ""),
            "category": tag["category"],
            "confidence": tag["confidence"],
            "is_critical": is_critical,
        })

    # Log critical flags
    critical_count = sum(1 for r in result if r["is_critical"])
    if critical_count > 0:
        log.warning(
            "⚠️  %d CRITICAL probe(s) detected (suicidal_ideation) — "
            "flag for clinical review.",
            critical_count,
        )

    log.info(
        "Probe detection complete: %d/%d interviewer turns tagged.",
        len(result),
        len(interviewer_turns),
    )
    return result


def tag_segments_with_probes(
    segments: list[dict[str, Any]],
    probe_tags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge probe category tags back into the full segment list.

    Each segment gets a ``probe_category`` key. Interviewer turns are tagged
    with their detected category. Subject (participant) responses inherit
    the category of the most recent preceding interviewer question, so
    downstream analysis can group responses by clinical topic.

    Parameters
    ----------
    segments : list[dict]
        Full transcript segments (all speakers).
    probe_tags : list[dict]
        Output from ``detect_probes()``.

    Returns
    -------
    list[dict]
        Deep-copied segments with ``probe_category``, ``probe_confidence``,
        and ``is_critical_probe`` keys added.
    """
    if not probe_tags:
        enriched = deepcopy(segments)
        for seg in enriched:
            seg["probe_category"] = None
            seg["probe_confidence"] = None
            seg["is_critical_probe"] = False
        return enriched

    # Build a lookup: (start, end) -> probe tag for fast matching
    probe_lookup: dict[tuple[float, float], dict[str, Any]] = {}
    for tag in probe_tags:
        key = (tag["start"], tag["end"])
        probe_lookup[key] = tag

    enriched: list[dict[str, Any]] = []
    current_category: Optional[str] = None
    current_confidence: Optional[float] = None
    current_critical: bool = False

    for seg in segments:
        seg_copy = deepcopy(seg)
        key = (seg.get("start", -1), seg.get("end", -1))

        if key in probe_lookup:
            # This is a tagged interviewer turn
            tag = probe_lookup[key]
            current_category = tag["category"]
            current_confidence = tag["confidence"]
            current_critical = tag.get("is_critical", False)

            seg_copy["probe_category"] = current_category
            seg_copy["probe_confidence"] = current_confidence
            seg_copy["is_critical_probe"] = current_critical
        else:
            # Subject or untagged turn — inherit from most recent probe
            seg_copy["probe_category"] = current_category
            seg_copy["probe_confidence"] = current_confidence
            seg_copy["is_critical_probe"] = current_critical

        enriched.append(seg_copy)

    return enriched


def summarize_probe_coverage(
    tagged_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize how well the clinical interview covered each probe category.

    Parameters
    ----------
    tagged_segments : list[dict]
        Output from ``tag_segments_with_probes()``.

    Returns
    -------
    dict
        Summary with keys:
        - ``categories``: per-category stats (question_count, response_count,
          total_duration_seconds, avg_confidence)
        - ``total_questions``: total interviewer probes detected
        - ``total_responses``: total subject responses across all categories
        - ``missing_categories``: list of categories never asked about
        - ``critical_flags``: list of categories flagged as critical that
          appeared (or were missing)
        - ``coverage_pct``: percentage of defined categories that were
          covered at least once (excluding "other" and "follow_up")
    """
    # Categories we consider clinically meaningful for coverage scoring
    CLINICAL_CATEGORIES = _VALID_CATEGORIES - {"other", "follow_up", "rapport_building"}

    cat_stats: dict[str, dict[str, Any]] = {}
    for name in CLINICAL_PROBE_CATEGORIES:
        cat_stats[name] = {
            "question_count": 0,
            "response_count": 0,
            "total_duration_seconds": 0.0,
            "confidences": [],
        }

    total_questions = 0
    total_responses = 0

    for seg in tagged_segments:
        cat = seg.get("probe_category")
        if cat is None or cat not in cat_stats:
            continue

        duration = max(0.0, seg.get("end", 0) - seg.get("start", 0))
        cat_stats[cat]["total_duration_seconds"] += duration

        # Determine if this segment is an interviewer turn (has confidence
        # from direct tagging) vs. an inherited subject response.
        # Interviewer turns have probe_confidence set from their own tag,
        # but we need to distinguish — we check if this segment's
        # (start, end) matches a probe_tag source. As a heuristic: if the
        # segment text starts with the interviewer probe text patterns or
        # the confidence was directly assigned (not inherited), we count
        # it as a question. Simpler: count segments whose category was
        # directly assigned (interviewer) vs. inherited (subject).
        # We re-use the fact that interviewer turns will have their exact
        # text in probe_tags and subject turns will not. Since we don't
        # have the original probe_tags here, we use a marker: interviewer
        # turns have is_critical_probe set precisely from the category,
        # which is also set for inherited segments. Instead, we check for
        # a reliable signal: the speaker role convention or segment text.
        #
        # The cleanest approach: check if the segment itself contributed
        # a probe question or is a response. We can infer this from
        # whether consecutive segments share a category — but the simplest
        # reliable method is to see if probe_confidence matches a direct
        # assignment. We'll count each category transition as a question
        # boundary.
        pass

    # Second pass: count questions vs responses by tracking category changes
    prev_category: Optional[str] = None
    in_question = False

    for seg in tagged_segments:
        cat = seg.get("probe_category")
        if cat is None or cat not in cat_stats:
            prev_category = None
            continue

        # A new category boundary or the first segment with this category
        # after a None/different category signals a new question
        conf = seg.get("probe_confidence")

        # Heuristic: if this segment is directly from the probe tags
        # (interviewer turn), it typically appears first when category
        # changes. We count it as a question if the category just changed
        # or if it's the very first segment of a category run.
        if cat != prev_category:
            cat_stats[cat]["question_count"] += 1
            total_questions += 1
            in_question = True
        elif in_question:
            # Same category, following the initial question turn
            # This is a response segment
            cat_stats[cat]["response_count"] += 1
            total_responses += 1
            in_question = False
        else:
            cat_stats[cat]["response_count"] += 1
            total_responses += 1

        if conf is not None:
            cat_stats[cat]["confidences"].append(conf)

        prev_category = cat

    # Finalize per-category summaries
    categories_summary: dict[str, dict[str, Any]] = {}
    for name, stats in cat_stats.items():
        confs = stats.pop("confidences")
        avg_conf = round(sum(confs) / len(confs), 3) if confs else None
        stats["total_duration_seconds"] = round(stats["total_duration_seconds"], 2)
        stats["avg_confidence"] = avg_conf
        categories_summary[name] = stats

    # Missing clinical categories (never asked)
    covered_clinical = {
        name
        for name in CLINICAL_CATEGORIES
        if categories_summary.get(name, {}).get("question_count", 0) > 0
    }
    missing = sorted(CLINICAL_CATEGORIES - covered_clinical)

    # Critical flags
    critical_flags: list[dict[str, Any]] = []
    for name, info in CLINICAL_PROBE_CATEGORIES.items():
        if info["critical"]:
            was_asked = categories_summary.get(name, {}).get("question_count", 0) > 0
            critical_flags.append({
                "category": name,
                "was_asked": was_asked,
                "question_count": categories_summary.get(name, {}).get("question_count", 0),
            })
            if not was_asked:
                log.warning(
                    "⚠️  Critical probe category '%s' was NOT covered in this interview.",
                    name,
                )

    # Coverage percentage
    coverage_pct = (
        round(len(covered_clinical) / len(CLINICAL_CATEGORIES) * 100, 1)
        if CLINICAL_CATEGORIES
        else 0.0
    )

    return {
        "categories": categories_summary,
        "total_questions": total_questions,
        "total_responses": total_responses,
        "missing_categories": missing,
        "critical_flags": critical_flags,
        "coverage_pct": coverage_pct,
    }


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def _load_question_detection_config(
    config_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load ClinicalWhisper config with question_detection section defaults."""
    cfg = load_config(config_path)
    qd = cfg.get("question_detection", {})

    defaults = {
        "enabled": True,
        "ollama_model": "qwen2:7b",
        "ollama_base_url": "http://localhost:11434",
        "timeout_seconds": 180,
        "max_retries": 1,
    }
    for key, default in defaults.items():
        qd.setdefault(key, default)

    cfg["question_detection"] = qd
    return cfg


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [QuestionDetector] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Question Detector — categorize clinical interview probes "
            "using a local Ollama LLM."
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="SEGMENTS_JSON",
        help=(
            "Path to a JSON file containing segments "
            '(list of {speaker, start, end, text} dicts)'
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the Ollama model (e.g. --model llama3:8b)",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help=(
            'JSON string mapping speakers to roles, e.g. '
            '\'{"Speaker 1": "interviewer", "Speaker 2": "subject"}\''
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent to Ollama without calling it.",
    )
    args = parser.parse_args()

    # Load config
    cfg = _load_question_detection_config()
    if args.model:
        cfg["question_detection"]["ollama_model"] = args.model

    # Use sample data if no file provided
    if args.file:
        file_path = Path(args.file).expanduser().resolve()
        if not file_path.exists():
            log.error("File not found: %s", file_path)
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        log.info("No --file provided, using built-in sample data.")
        segments = [
            {"speaker": "Ellie", "start": 0.0, "end": 3.5, "text": "Hi, how are you doing today?"},
            {"speaker": "Participant", "start": 3.5, "end": 6.0, "text": "I'm doing okay, I guess."},
            {"speaker": "Ellie", "start": 6.0, "end": 10.0, "text": "What are some things that make you really happy?"},
            {"speaker": "Participant", "start": 10.0, "end": 18.0, "text": "I like spending time with my dog. We go to the park."},
            {"speaker": "Ellie", "start": 18.0, "end": 22.0, "text": "That sounds nice. Tell me more about that."},
            {"speaker": "Participant", "start": 22.0, "end": 30.0, "text": "Yeah my dog's name is Max, he's a golden retriever."},
            {"speaker": "Ellie", "start": 30.0, "end": 36.0, "text": "Have you been feeling sad or down lately?"},
            {"speaker": "Participant", "start": 36.0, "end": 45.0, "text": "Yeah, sometimes I feel pretty low. It's been hard."},
            {"speaker": "Ellie", "start": 45.0, "end": 50.0, "text": "How has your sleep been?"},
            {"speaker": "Participant", "start": 50.0, "end": 58.0, "text": "Not great. I wake up a lot during the night."},
            {"speaker": "Ellie", "start": 58.0, "end": 65.0, "text": "Have you ever experienced any traumatic events in your life?"},
            {"speaker": "Participant", "start": 65.0, "end": 75.0, "text": "I was in a car accident two years ago. I still think about it."},
            {"speaker": "Ellie", "start": 75.0, "end": 82.0, "text": "Do you spend much time with friends or family?"},
            {"speaker": "Participant", "start": 82.0, "end": 90.0, "text": "Not really. I've kind of pulled away from everyone."},
            {"speaker": "Ellie", "start": 90.0, "end": 96.0, "text": "What do you do when you're feeling stressed out?"},
            {"speaker": "Participant", "start": 96.0, "end": 105.0, "text": "I just try to sleep or watch TV. Nothing really helps."},
            {"speaker": "Ellie", "start": 105.0, "end": 112.0, "text": "Have you ever had thoughts of hurting yourself?"},
            {"speaker": "Participant", "start": 112.0, "end": 120.0, "text": "Sometimes, yeah. But I wouldn't act on it."},
            {"speaker": "Ellie", "start": 120.0, "end": 125.0, "text": "Thank you for sharing that with me. We're done for today."},
        ]

    # Parse roles
    if args.roles:
        try:
            roles = json.loads(args.roles)
        except json.JSONDecodeError as exc:
            log.error("Invalid --roles JSON: %s", exc)
            sys.exit(1)
    else:
        # Auto-detect: assume the first speaker is the interviewer
        speakers_seen: list[str] = []
        for seg in segments:
            spk = seg.get("speaker", "")
            if spk and spk not in speakers_seen:
                speakers_seen.append(spk)
        if len(speakers_seen) >= 2:
            roles = {speakers_seen[0]: "interviewer", speakers_seen[1]: "subject"}
        elif len(speakers_seen) == 1:
            roles = {speakers_seen[0]: "interviewer"}
        else:
            roles = {}
        log.info("Auto-detected roles: %s", roles)

    if args.dry_run:
        # Show the prompt that would be sent
        interviewer_speakers = {
            spk for spk, role in roles.items()
            if role.lower() in ("interviewer", "clinician", "therapist", "ellie")
        }
        turns = [s for s in segments if s.get("speaker") in interviewer_speakers and s.get("text", "").strip()]
        prompt = QUESTION_DETECTION_PROMPT.format(
            category_block=_build_category_block(),
            turns_block=_build_turns_block(turns),
        )
        print("=" * 60)
        print("DRY RUN — Prompt that would be sent to Ollama:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"Interviewer turns: {len(turns)}")
        print(f"Model: {cfg['question_detection']['ollama_model']}")
        sys.exit(0)

    # Run probe detection
    print("=" * 60)
    print("QUESTION DETECTOR — Clinical Probe Categorization")
    print("=" * 60)

    probe_tags = detect_probes(segments, roles, cfg)

    if not probe_tags:
        print("\nNo probes detected (check logs for details).")
        sys.exit(0)

    print(f"\n{'IDX':>4}  {'TIME':>12}  {'CATEGORY':<22}  {'CONF':>5}  TEXT")
    print("-" * 90)
    for i, tag in enumerate(probe_tags):
        time_range = f"{_fmt_time(tag['start'])}-{_fmt_time(tag['end'])}"
        critical_marker = " ⚠️" if tag["is_critical"] else ""
        text_preview = tag["text"][:50] + ("..." if len(tag["text"]) > 50 else "")
        print(
            f"{i:>4}  {time_range:>12}  {tag['category']:<22}  {tag['confidence']:>5.2f}  "
            f"{text_preview}{critical_marker}"
        )

    # Tag all segments
    tagged = tag_segments_with_probes(segments, probe_tags)
    print(f"\nTagged {len(tagged)} total segments with probe categories.")

    # Coverage summary
    summary = summarize_probe_coverage(tagged)
    print(f"\n{'=' * 60}")
    print("PROBE COVERAGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total questions:  {summary['total_questions']}")
    print(f"Total responses:  {summary['total_responses']}")
    print(f"Coverage:         {summary['coverage_pct']}%")

    print(f"\n{'CATEGORY':<22}  {'Q':>3}  {'R':>3}  {'DURATION':>10}  {'CONF':>6}")
    print("-" * 55)
    for cat_name, stats in summary["categories"].items():
        if stats["question_count"] == 0 and stats["response_count"] == 0:
            continue
        dur = _fmt_time(stats["total_duration_seconds"])
        conf_str = f"{stats['avg_confidence']:.2f}" if stats["avg_confidence"] is not None else "  n/a"
        print(f"{cat_name:<22}  {stats['question_count']:>3}  {stats['response_count']:>3}  {dur:>10}  {conf_str:>6}")

    if summary["missing_categories"]:
        print(f"\n⚠️  Missing clinical categories: {', '.join(summary['missing_categories'])}")

    for flag in summary["critical_flags"]:
        status = "✅ asked" if flag["was_asked"] else "❌ NOT asked"
        print(f"   Critical probe '{flag['category']}': {status} ({flag['question_count']} questions)")
