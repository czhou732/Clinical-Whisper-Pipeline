#!/usr/bin/env python3
"""
LLM Clinical Scorer — ClinicalWhisper
======================================
Uses a local LLM (via Ollama) to produce structured clinical assessments
from formatted clinical interview transcripts.

Sends the transcript + acoustic context to the local Ollama endpoint and
parses the JSON response into a validated scoring dictionary.

All processing is local — no data leaves the machine.

Config section (config.yaml):
-----------------------------
  llm_scoring:
    enabled: true
    ollama_model: 'qwen2:7b'
    ollama_base_url: 'http://localhost:11434'
    timeout_seconds: 300
    max_retries: 1

Usage:
  # Score a transcript file:
  python llm_clinical_scorer.py --file /path/to/transcript.txt

  # With acoustic context from a JSON file:
  python llm_clinical_scorer.py --file transcript.txt --acoustics acoustics.json

  # Override model:
  python llm_clinical_scorer.py --file transcript.txt --model llama3:8b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    mx = None
    load = None
    generate = None

try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None

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
# Constants
# ---------------------------------------------------------------------------
OLLAMA_GENERATE_ENDPOINT = "/api/generate"

REQUIRED_SCORE_KEYS = [
    "hesitancy_score",
    "affect_flatness",
    "engagement_level",
    "elaboration_positive",
    "elaboration_negative",
    "psychomotor_indicators",
]

DEFAULT_SCORES: dict[str, Any] = {
    "hesitancy_score": 0,
    "affect_flatness": 0,
    "engagement_level": 5,
    "elaboration_positive": 5,
    "elaboration_negative": 5,
    "psychomotor_indicators": 0,
    "key_observations": [],
    "clinical_impression": "Scoring could not be completed.",
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
CLINICAL_SCORING_PROMPT = """\
You are a clinical research assistant analyzing a structured interview transcript. \
Score each dimension 0-10 based on evidence in the transcript. \
Cite specific quotes to support each score.

Your task is to analyze the following clinical interview transcript and optional \
acoustic/prosodic data, then produce a structured clinical assessment as a JSON object.

## Scoring Dimensions (each 0-10)

- **hesitancy_score**: How much the subject hesitates, pauses, or uses filler words \
(um, uh, like, you know, long pauses marked as [...], false starts, self-corrections). \
0 = fluent and decisive, 10 = extremely hesitant with constant fillers and restarts.

- **affect_flatness**: How emotionally flat or blunted the subject's responses are. \
Look for monotone descriptions, lack of emotional language, absence of affective words, \
minimal variation in expression. \
0 = rich emotional expression, 10 = completely flat/blunted affect.

- **engagement_level**: How engaged and interactive the subject is with the interviewer. \
Look for question-asking, elaboration beyond what is asked, humor, topic initiation, \
responsive follow-ups vs. monosyllabic answers. \
0 = completely disengaged/monosyllabic, 10 = highly engaged and interactive.

- **elaboration_positive**: How much the subject elaborates when discussing positive \
topics (enjoyable activities, achievements, relationships, future plans). \
0 = no elaboration on positive topics, 10 = extensive positive elaboration.

- **elaboration_negative**: How much the subject elaborates when discussing negative \
topics (problems, distress, losses, complaints, symptoms). \
0 = no elaboration on negative topics, 10 = extensive negative elaboration.

- **psychomotor_indicators**: Signs of psychomotor retardation or agitation in speech \
patterns — unusually slow responses, trailing off, pressured speech, abrupt topic \
changes, or marked latency. \
0 = normal speech rhythm, 10 = severe psychomotor disturbance.

## Output Format

Return a single JSON object with exactly these keys:

```json
{{
  "hesitancy_score": <0-10>,
  "affect_flatness": <0-10>,
  "engagement_level": <0-10>,
  "elaboration_positive": <0-10>,
  "elaboration_negative": <0-10>,
  "psychomotor_indicators": <0-10>,
  "key_observations": [
    "observation 1 with 'quoted evidence'",
    "observation 2 with 'quoted evidence'"
  ],
  "clinical_impression": "One paragraph summarizing the clinical picture."
}}
```

## Few-Shot Examples

### Example 1 — Mildly Disengaged Subject

**Transcript excerpt:**
Interviewer: How have you been spending your time lately?
Subject: Um... I don't know. Just, like, the usual stuff I guess.
Interviewer: Can you tell me more about that?
Subject: Not really. Just... hanging around.
Interviewer: Have you been enjoying anything recently?
Subject: Not really, no.

**Acoustic context:** pitch_cv: 0.08, loudness_cv: 0.05, vta: 4.2

**Correct output:**
```json
{{
  "hesitancy_score": 6,
  "affect_flatness": 7,
  "engagement_level": 2,
  "elaboration_positive": 1,
  "elaboration_negative": 2,
  "psychomotor_indicators": 4,
  "key_observations": [
    "Subject uses frequent fillers: 'Um... I don't know. Just, like, the usual stuff'",
    "Minimal elaboration on any topic — responses are monosyllabic or near-monosyllabic",
    "No positive content volunteered — 'Not really, no' when asked about enjoyment",
    "Low pitch variability (pitch_cv: 0.08) supports flat affect observation"
  ],
  "clinical_impression": "The subject presents with markedly reduced engagement and flat affect. Responses are vague and minimal, with frequent fillers suggesting either cognitive sluggishness or reluctance to engage. There is a notable absence of any positive elaboration and low prosodic variability, consistent with anhedonic or depressive presentation. Psychomotor slowing is mildly suggested by the trailing responses and pauses."
}}
```

### Example 2 — Engaged Subject With Selective Negativity

**Transcript excerpt:**
Interviewer: Tell me about your week.
Subject: It's been okay, actually. I went to my daughter's soccer game on Saturday which was great — she scored two goals! I was really proud of her.
Interviewer: That sounds wonderful. Anything difficult this week?
Subject: Yeah, work has been really stressful. My boss has been on my case about this project and I just feel like nothing I do is good enough. I stayed late three nights this week and it's exhausting. I barely see my kids during the week now.

**Acoustic context:** pitch_cv: 0.22, loudness_cv: 0.18, vta: 1.8

**Correct output:**
```json
{{
  "hesitancy_score": 1,
  "affect_flatness": 1,
  "engagement_level": 9,
  "elaboration_positive": 7,
  "elaboration_negative": 8,
  "psychomotor_indicators": 1,
  "key_observations": [
    "Subject is fluent and articulate with no fillers or pauses",
    "Rich positive elaboration: 'she scored two goals! I was really proud of her'",
    "Extensive negative elaboration with specific details: 'stayed late three nights', 'barely see my kids'",
    "Strong emotional language: 'really proud', 'really stressful', 'exhausting'",
    "High prosodic variability (pitch_cv: 0.22) consistent with expressive speech"
  ],
  "clinical_impression": "The subject is highly engaged and emotionally expressive across both positive and negative domains. They provide detailed, specific accounts of experiences and demonstrate a full range of affect. Speech is fluent and well-organized. The notable finding is selective distress around work demands with preserved capacity for positive experience, suggesting situational stress rather than a pervasive mood disturbance."
}}
```

## Now analyze the following:

### Interview Transcript
{transcript}

### Acoustic Context
{acoustic_context}

Return ONLY the JSON object, no additional text before or after it.
"""


# ===================================================================
# Config
# ===================================================================

def _load_scoring_config(
    config_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load ClinicalWhisper config and fill llm_scoring section defaults."""
    cfg = load_config(config_path)
    sc = cfg.get("llm_scoring", {})

    defaults = {
        "enabled": True,
        "mlx_model": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "hf_model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "max_tokens": 1200,
        "timeout_seconds": 300,
        "max_retries": 1,
    }
    for key, default in defaults.items():
        sc.setdefault(key, default)

    cfg["llm_scoring"] = sc
    return cfg


# ===================================================================
# MLX integration
# ===================================================================

_MODEL_CACHE = {}

def call_local_lm(
    prompt: str,
    model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    timeout: int = 300,
    hf_model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
    max_tokens: int = 1200,
) -> str:
    """
    Generate text using a local LLM. Tries mlx-lm first (Apple Silicon),
    falls back to HuggingFace transformers (Windows/Linux/Intel Mac).
    """
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Try MLX first
    if load is not None and sys.platform == "darwin":
        log.info(f"Loading MLX model {model_name}...")
        
        if model_name not in _MODEL_CACHE:
            model, tokenizer = load(model_name)
            _MODEL_CACHE[model_name] = (model, tokenizer)
        else:
            model, tokenizer = _MODEL_CACHE[model_name]

        log.info("Generating response with MLX...")
        
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens,
            verbose=False
        )
        return response

    # Fallback to Transformers
    log.info("MLX not available or not on Apple Silicon. Falling back to Transformers...")
    if pipeline is None:
        raise RuntimeError("Transformers library is not installed. Please pip install transformers torch")
        
    # NousResearch mirrors Llama-3-8B-Instruct without the gated-repo licence
    # click-through, so Windows/Linux users need no Hugging Face account.
    log.info(f"Loading HF model {hf_model_name}...")

    if hf_model_name not in _MODEL_CACHE:
        device_map = "auto" if torch.cuda.is_available() else None
        generator = pipeline(
            "text-generation",
            model=hf_model_name,
            device_map=device_map,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        _MODEL_CACHE[hf_model_name] = generator
    else:
        generator = _MODEL_CACHE[hf_model_name]

    log.info("Generating response with Transformers...")
    outputs = generator(formatted_prompt, max_new_tokens=max_tokens, return_full_text=False)
    return outputs[0]["generated_text"]


# ===================================================================
# Response parsing
# ===================================================================

def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` object in *text*, or None.

    Brace counting is string-aware so that braces inside quoted values (and
    escaped quotes) do not throw off the depth.

    If the text ends before the object closes — which happens whenever the
    token cap truncates a long ``key_observations`` list — the open containers
    are closed so the scores that *did* arrive are still usable. A truncated
    response is far more useful repaired than discarded, because discarding it
    silently substitutes default scores.
    """
    # Chat special tokens mark where the model stopped answering and started
    # a fresh turn. Everything after the first one is not part of the object.
    cut = re.search(r"<\|(?:eot_id|start_header_id|end_of_text)\|>", text)
    if cut:
        text = text[: cut.start()]

    start = text.find("{")
    if start == -1:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    last_value_end = -1  # last index at which a complete value could end

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                last_value_end = i
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_value_end = i
            if not stack:
                return text[start:i + 1]
        elif ch.isdigit() or ch in "eE+-.":
            last_value_end = i

    # Ran out of text with containers still open. Llama-3 routinely emits the
    # whole body and then stops without the final "}", so close what is open
    # rather than discarding scores that did arrive.
    return _repair_truncated_json(text, start, stack, in_string, last_value_end)


def _repair_truncated_json(
    text: str,
    start: int,
    stack: list[str],
    in_string: bool,
    last_value_end: int,
) -> Optional[str]:
    """Close an unterminated JSON object so the complete keys survive."""
    if not stack:
        return None

    if in_string:
        # Cut back to the last complete value; a half-written string is not
        # worth guessing at.
        if last_value_end <= start:
            return None
        fragment = text[start:last_value_end + 1]
    else:
        fragment = text[start:max(last_value_end, start) + 1]

    # Drop a dangling trailing comma, or an orphaned key left by the cut —
    # either `"key":` with no value, or a bare `"key"` with no colon yet.
    fragment = re.sub(r",\s*$", "", fragment.rstrip())
    fragment = re.sub(r",?\s*\"[^\"]*\"\s*:\s*$", "", fragment.rstrip())
    fragment = re.sub(r",\s*\"[^\"]*\"\s*$", "", fragment.rstrip())

    for opener in reversed(stack):
        fragment += "]" if opener == "[" else "}"

    try:
        json.loads(fragment)
    except json.JSONDecodeError:
        return None

    log.warning(
        "LLM response did not close its JSON object; repaired it to recover "
        "the scores. Raise llm_scoring.max_tokens if this recurs."
    )
    return fragment


def parse_scoring_response(raw: str) -> dict[str, Any]:
    """
    Extract and validate a clinical scoring JSON object from raw LLM output.

    Handles common LLM output quirks:
    - Markdown code fences (```json ... ```)
    - Extra text before/after the JSON block
    - Missing keys (filled with defaults)
    - Out-of-range scores (clamped to 0-10)

    Parameters
    ----------
    raw : str
        The raw text response from the LLM.

    Returns
    -------
    dict
        Validated scoring dictionary with all required keys.

    Raises
    ------
    ValueError
        If no valid JSON object can be extracted from the response.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()

    # Try to extract from ```json ... ``` blocks first
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        cleaned,
        re.DOTALL,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Take the FIRST balanced object. Spanning first "{" to last "}" swallowed
    # any commentary or second object the model appended after its answer,
    # which then failed to parse and silently fell back to default scores.
    json_str = _first_json_object(cleaned)

    if json_str is None:
        raise ValueError(
            f"No JSON object found in LLM response. "
            f"Preview: {raw[:300]!r}"
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in LLM response: {exc}. "
            f"Preview: {json_str[:400]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"LLM response parsed to {type(data).__name__}, expected dict."
        )

    # Fill missing score keys with defaults
    for key in REQUIRED_SCORE_KEYS:
        if key not in data:
            log.warning("Missing score key '%s', using default: %s", key, DEFAULT_SCORES[key])
            data[key] = DEFAULT_SCORES[key]

    # Clamp all numeric scores to 0-10
    for key in REQUIRED_SCORE_KEYS:
        val = data[key]
        if isinstance(val, (int, float)):
            clamped = max(0, min(10, val))
            if clamped != val:
                log.warning(
                    "Score '%s' was %s, clamped to %s", key, val, clamped
                )
            data[key] = clamped
        else:
            log.warning(
                "Score '%s' has non-numeric value %r, using default", key, val
            )
            data[key] = DEFAULT_SCORES[key]

    # Ensure key_observations is a list of strings
    if "key_observations" not in data or not isinstance(data["key_observations"], list):
        log.warning("Missing or invalid 'key_observations', using empty list")
        data["key_observations"] = []
    else:
        # Filter out any non-string entries
        data["key_observations"] = [
            str(obs) for obs in data["key_observations"] if obs
        ]

    # Ensure clinical_impression is a string
    if "clinical_impression" not in data or not isinstance(data["clinical_impression"], str):
        log.warning("Missing or invalid 'clinical_impression', using default")
        data["clinical_impression"] = DEFAULT_SCORES["clinical_impression"]

    return data


# ===================================================================
# Main scoring function
# ===================================================================

def score_transcript(
    structured_transcript: str,
    acoustic_context: str = "No acoustic data available.",
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Score a clinical interview transcript using a local LLM.

    Builds the full prompt from the template, transcript, and acoustic
    context, sends it to Ollama, and returns a validated scoring dict.

    Parameters
    ----------
    structured_transcript : str
        The interview transcript in Interviewer:/Subject: format.
    acoustic_context : str
        Serialized prosodic features (e.g. from AcousticExtractor).
        Can be a JSON string or human-readable summary.
    config : dict, optional
        Full ClinicalWhisper config dict. If None, loads from default path.

    Returns
    -------
    dict
        Scoring dictionary with keys:
        - hesitancy_score (0-10)
        - affect_flatness (0-10)
        - engagement_level (0-10)
        - elaboration_positive (0-10)
        - elaboration_negative (0-10)
        - psychomotor_indicators (0-10)
        - key_observations (list[str])
        - clinical_impression (str)
        - _meta (dict with model, elapsed_seconds, raw_response_length)
    """
    if config is None:
        config = _load_scoring_config()

    sc_cfg = config.get("llm_scoring", {})
    model = sc_cfg.get("mlx_model", "mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    hf_model = sc_cfg.get("hf_model", "NousResearch/Meta-Llama-3-8B-Instruct")
    max_tokens = sc_cfg.get("max_tokens", 1200)
    timeout = sc_cfg.get("timeout_seconds", 300)

    # Build the full prompt
    prompt = CLINICAL_SCORING_PROMPT.format(
        transcript=structured_transcript,
        acoustic_context=acoustic_context,
    )

    # Truncate very long transcripts to stay within context window
    words = prompt.split()
    if len(words) > 6000:
        log.warning(
            "Prompt is %d words; truncating transcript to fit context window.",
            len(words),
        )
        # Re-build with truncated transcript
        transcript_words = structured_transcript.split()
        truncated = " ".join(transcript_words[:3000]) + "\n[...truncated...]"
        prompt = CLINICAL_SCORING_PROMPT.format(
            transcript=truncated,
            acoustic_context=acoustic_context,
        )

    # Call MLX
    t0 = time.time()
    try:
        raw_response = call_local_lm(
            prompt=prompt,
            model_name=model,
            timeout=timeout,
            hf_model_name=hf_model,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        log.error("LLM generation failed: %s", exc)
        result = dict(DEFAULT_SCORES)
        result["_meta"] = {
            "model": model,
            "elapsed_seconds": round(time.time() - t0, 1),
            "error": str(exc),
        }
        return result

    elapsed = round(time.time() - t0, 1)

    # Parse and validate the response
    try:
        scoring = parse_scoring_response(raw_response)
    except ValueError as exc:
        log.error("Failed to parse LLM scoring response: %s", exc)
        log.debug("Raw response: %s", raw_response[:500])
        scoring = dict(DEFAULT_SCORES)
        scoring["_parse_error"] = str(exc)

    # Attach metadata
    scoring["_meta"] = {
        "model": model,
        "elapsed_seconds": elapsed,
        "raw_response_length": len(raw_response),
    }

    return scoring


# ===================================================================
# CLI entry point
# ===================================================================

def _format_scoring_output(scoring: dict[str, Any]) -> str:
    """Format a scoring dict as a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  CLINICAL INTERVIEW SCORING — LLM Assessment")
    lines.append("=" * 60)
    lines.append("")

    # Score bars
    score_labels = {
        "hesitancy_score": "Hesitancy",
        "affect_flatness": "Affect Flatness",
        "engagement_level": "Engagement",
        "elaboration_positive": "Elaboration (Positive)",
        "elaboration_negative": "Elaboration (Negative)",
        "psychomotor_indicators": "Psychomotor Indicators",
    }

    for key, label in score_labels.items():
        val = scoring.get(key, 0)
        if isinstance(val, (int, float)):
            val_int = int(round(val))
            bar = "\u2588" * val_int + "\u2591" * (10 - val_int)
            lines.append(f"  {label:<26s} {bar} {val_int}/10")
        else:
            lines.append(f"  {label:<26s} N/A")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  Key Observations:")
    lines.append("-" * 60)
    observations = scoring.get("key_observations", [])
    if observations:
        for obs in observations:
            lines.append(f"  \u2022 {obs}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  Clinical Impression:")
    lines.append("-" * 60)
    impression = scoring.get("clinical_impression", "N/A")
    # Word-wrap impression at ~72 chars
    words = impression.split()
    current_line = "  "
    for word in words:
        if len(current_line) + len(word) + 1 > 72:
            lines.append(current_line)
            current_line = "  " + word
        else:
            current_line += (" " if current_line.strip() else "") + word
    if current_line.strip():
        lines.append(current_line)

    lines.append("")

    # Metadata
    meta = scoring.get("_meta", {})
    if meta:
        lines.append("-" * 60)
        lines.append(f"  Model: {meta.get('model', 'unknown')}")
        lines.append(f"  Inference time: {meta.get('elapsed_seconds', '?')}s")
        if "error" in meta:
            lines.append(f"  Error: {meta['error']}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for testing the clinical scorer."""
    parser = argparse.ArgumentParser(
        description=(
            "LLM Clinical Scorer — score a clinical interview transcript "
            "using a local Ollama LLM"
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        metavar="TRANSCRIPT",
        help="Path to the transcript text file (Interviewer:/Subject: format)",
    )
    parser.add_argument(
        "--acoustics",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Optional path to acoustic features JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the Ollama model (e.g. --model llama3:8b)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    # Set up logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ClinicalScorer] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load transcript
    transcript_path = Path(args.file).expanduser().resolve()
    if not transcript_path.exists():
        log.error("Transcript file not found: %s", transcript_path)
        sys.exit(1)

    transcript_text = transcript_path.read_text(encoding="utf-8")
    if not transcript_text.strip():
        log.error("Transcript file is empty: %s", transcript_path)
        sys.exit(1)

    log.info("Loaded transcript: %s (%d chars)", transcript_path.name, len(transcript_text))

    # Load acoustic context if provided
    acoustic_context = "No acoustic data available."
    if args.acoustics:
        acoustics_path = Path(args.acoustics).expanduser().resolve()
        if acoustics_path.exists():
            raw_acoustics = acoustics_path.read_text(encoding="utf-8")
            try:
                acoustic_data = json.loads(raw_acoustics)
                acoustic_context = json.dumps(acoustic_data, indent=2)
                log.info("Loaded acoustic context: %s", acoustics_path.name)
            except json.JSONDecodeError:
                # Treat as plain text
                acoustic_context = raw_acoustics
                log.info("Loaded acoustic context as text: %s", acoustics_path.name)
        else:
            log.warning("Acoustic file not found: %s", acoustics_path)

    # Load config and apply model override
    cfg = _load_scoring_config(args.config)
    if args.model:
        cfg["llm_scoring"]["ollama_model"] = args.model

    # Run scoring
    log.info("Starting clinical scoring...")
    scoring = score_transcript(
        structured_transcript=transcript_text,
        acoustic_context=acoustic_context,
        config=cfg,
    )

    # Output
    if args.json:
        print(json.dumps(scoring, indent=2, ensure_ascii=False))
    else:
        print(_format_scoring_output(scoring))


if __name__ == "__main__":
    main()
