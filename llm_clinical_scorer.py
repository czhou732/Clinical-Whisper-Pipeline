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
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    mx = None
    load = None
    generate = None
    stream_generate = None
    make_sampler = None

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
# Windowing and aggregation
# ---------------------------------------------------------------------------
# Llama-3 has an 8K context and the prompt template alone is ~2600 words, so a
# window of ~2800 transcript words is what actually fits.
WINDOW_WORDS = 2800


def filter_to_subject(transcript: str) -> str:
    """Keep only the Subject's turns.

    Optional, not the default. The prompt already instructs the model to score
    the subject, and engagement_level is an interactional judgement that needs
    the interviewer's questions to be meaningful — dropping them changes what
    that dimension measures. Offered for analyses that want the subject's
    language in isolation.
    """
    kept = [
        line for line in transcript.splitlines()
        if re.search(r"\]\s*Subject\s*:", line) or line.strip().startswith("Subject:")
    ]
    return "\n".join(kept) if kept else transcript


def _split_into_windows(transcript: str, window_words: int = WINDOW_WORDS) -> list[str]:
    """Split a transcript into scoreable windows, breaking on turn boundaries.

    Splitting mid-sentence would hand the model a fragment with no speaker
    attribution, so windows break between lines and a line longer than the
    window is emitted on its own rather than cut.
    """
    if not transcript.strip():
        return [transcript]

    lines = transcript.splitlines()
    windows: list[str] = []
    current: list[str] = []
    count = 0

    for line in lines:
        n = len(line.split())
        if current and count + n > window_words:
            windows.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n

    if current:
        windows.append("\n".join(current))

    return windows or [transcript]


def _aggregate_scores(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-window (and per-sample) scores into one result.

    The six dimensions are averaged across runs and rounded; the spread is
    reported alongside so a score that varied widely across an interview is
    visible rather than hidden behind a single number. Observations are pooled
    in order, de-duplicated, and the longest impression is kept as the summary.
    """
    out: dict[str, Any] = {}
    spread: dict[str, Any] = {}

    for key in REQUIRED_SCORE_KEYS:
        values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if not values:
            out[key] = DEFAULT_SCORES[key]
            continue
        mean = sum(values) / len(values)
        out[key] = int(round(mean))
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / len(values)
            spread[key] = {
                "mean": round(mean, 2),
                "sd": round(var ** 0.5, 2),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

    seen: set[str] = set()
    observations: list[str] = []
    for r in runs:
        for obs in r.get("key_observations") or []:
            if isinstance(obs, str) and obs not in seen:
                seen.add(obs)
                observations.append(obs)
    out["key_observations"] = observations[:12]

    impressions = [
        r.get("clinical_impression")
        for r in runs
        if isinstance(r.get("clinical_impression"), str) and r.get("clinical_impression")
    ]
    out["clinical_impression"] = (
        max(impressions, key=len) if impressions else DEFAULT_SCORES["clinical_impression"]
    )
    if len(impressions) > 1:
        out["window_impressions"] = impressions

    if spread:
        out["score_spread"] = spread

    return out


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
        "greedy_first": True,
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


def unload_models() -> int:
    """Drop the cached scoring model (~4.5 GB for Llama-3-8B at 4-bit)."""
    import gc

    freed = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    gc.collect()

    # MLX holds freed buffers in its own pool; hand them back explicitly.
    try:
        if mx is not None:
            mx.clear_cache()
    except Exception:  # pragma: no cover - best effort
        pass

    if freed:
        log.info("Released clinical scoring model from memory.")
    return freed

def _llama_prompt(prompt: str) -> str:
    return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def _answer_complete(text: str) -> bool:
    """True once the answer holds every score and the closing impression.

    clinical_impression is the last field, so nothing is tested before it
    appears; otherwise a repair could close the object after the first
    observation and truncate the answer.
    """
    if '"clinical_impression"' not in text:
        return False
    candidate = _first_json_object(text, quiet=True)
    if not candidate:
        return False
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return all(k in data for k in REQUIRED_SCORE_KEYS) and bool(data.get("clinical_impression"))


def _mlx_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        log.info(f"Loading MLX model {model_name}...")
        _MODEL_CACHE[model_name] = load(model_name)
    return _MODEL_CACHE[model_name]


def call_local_lm_reusing_prompt(
    prompt: str,
    temperatures: list[float],
    model_name: str,
    max_tokens: int = 1200,
    should_cancel=None,
    on_pass=None,
) -> list[str]:
    """Several scoring passes over one chunk, reading the chunk only once.

    Measured on a 6,030-token chunk: reading the prompt takes 10.4 s of a
    12.6 s pass, writing the answer 2.3 s. Re-reading an identical chunk for
    each of the 5 passes is the cost, so the model reads it once, and after
    each pass its memory is rewound to the end of the prompt for the next.
    Each pass is otherwise the same as before: same sampling, same early stop.
    """
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    model, tokenizer = _mlx_model(model_name)
    text = _llama_prompt(prompt)
    add_special = tokenizer.bos_token is None or not text.startswith(tokenizer.bos_token)
    tokens = tokenizer.encode(text, add_special_tokens=add_special)

    # Read everything but the last token, in the same 2,048-token pieces
    # mlx-lm uses, then let each pass start from the last token.
    cache = make_prompt_cache(model)
    prefix = tokens[:-1]
    for i in range(0, len(prefix), 2048):
        model(mx.array(prefix[i:i + 2048])[None], cache=cache)
        mx.eval([c.state for c in cache])
    read_to = cache[0].offset

    answers = []
    for s_idx, temp in enumerate(temperatures):
        if should_cancel is not None and should_cancel():
            raise ScoringCancelled("Scoring stopped.")
        kwargs = {"sampler": make_sampler(temp=temp)} if temp > 0 else {}
        chunks: list[str] = []
        for n, part in enumerate(stream_generate(
                model, tokenizer, prompt=mx.array(tokens[-1:]), max_tokens=max_tokens,
                prompt_cache=cache, **kwargs), start=1):
            chunks.append(part.text)
            if (n % 8 == 0 or '"}' in part.text) and _answer_complete("".join(chunks)):
                break
        answers.append("".join(chunks))
        trim_prompt_cache(cache, cache[0].offset - read_to)  # rewind for the next pass
        if on_pass is not None:
            on_pass(s_idx)
    return answers


def call_local_lm(
    prompt: str,
    model_name: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    timeout: int = 300,
    hf_model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
    max_tokens: int = 1200,
    temperature: float = 0.0,
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

        # Stop as soon as the JSON object closes. Left to run, Llama-3 writes
        # its ~1100-character answer and then keeps going — fresh assistant
        # turns, repeated chatter — until it hits the cap. The parser discards
        # that tail but it is paid for in full: roughly 80% of generation time,
        # and it is what pushed responses past the cap and left objects
        # unterminated in the first place.
        if stream_generate is not None:
            chunks: list[str] = []
            checks = 0
            stream_kwargs = {}
            if temperature > 0.0 and make_sampler is not None:
                # Greedy decoding is deterministic, so repeated runs would be
                # identical and report zero spread. Uncertainty estimates need
                # an actual sampler.
                stream_kwargs["sampler"] = make_sampler(temp=temperature)

            for part in stream_generate(
                model, tokenizer, prompt=formatted_prompt,
                max_tokens=max_tokens, **stream_kwargs
            ):
                chunks.append(part.text)

                # clinical_impression is the last field, so only start testing
                # once it appears — otherwise the repair would happily close
                # the object after the first observation and truncate the
                # answer. Waiting for a literal "}" does not work: this model
                # usually ends the impression string and emits EOS without ever
                # closing the object.
                checks += 1
                if checks % 8 and '"}' not in part.text:
                    continue
                buffer = "".join(chunks)
                if '"clinical_impression"' not in buffer:
                    continue
                candidate = _first_json_object(buffer, quiet=True)
                if not candidate:
                    continue
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if all(k in data for k in REQUIRED_SCORE_KEYS) and data.get(
                    "clinical_impression"
                ):
                    break
            return "".join(chunks)

        return generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

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

def _first_json_object(text: str, quiet: bool = False) -> Optional[str]:
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
    return _repair_truncated_json(text, start, stack, in_string, last_value_end, quiet)


def _repair_truncated_json(
    text: str,
    start: int,
    stack: list[str],
    in_string: bool,
    last_value_end: int,
    quiet: bool = False,
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

    if not quiet:
        # Silenced while streaming: the probe runs on every chunk, and this is
        # the expected shape of a Llama-3 answer, not an anomaly worth a line
        # per token in the user-facing console.
        log.debug("Repaired an unterminated JSON object from the LLM response.")
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
        # No preview of the answer in the message: it lands in the log, and the
        # model's answer quotes the (de-identified) transcript.
        raise ValueError(f"No JSON object found in LLM response ({len(raw)} characters).")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in LLM response: {exc}.") from exc

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

class ScoringCancelled(RuntimeError):
    """Stop was pressed during scoring."""


def score_transcript(
    structured_transcript: str,
    acoustic_context: str = "No acoustic data available.",
    config: Optional[dict[str, Any]] = None,
    progress=None,
    should_cancel=None,
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
    greedy_first = bool(sc_cfg.get("greedy_first", True))
    timeout = sc_cfg.get("timeout_seconds", 300)

    samples = max(1, int(sc_cfg.get("samples", 1)))
    temperature = float(sc_cfg.get("temperature", 0.0))
    window_words = int(sc_cfg.get("window_words", WINDOW_WORDS))

    # Windows, not truncation. The previous behaviour kept the first 3000 words
    # and dropped the rest silently, so a 60-minute interview was scored on its
    # first third while the output looked complete.
    scope = sc_cfg.get("transcript_scope", "dialogue")
    scored_transcript = structured_transcript
    if scope == "subject_only":
        scored_transcript = filter_to_subject(structured_transcript)
        log.info("Scoring the subject's turns only (transcript_scope: subject_only).")

    windows = _split_into_windows(scored_transcript, window_words)
    total_words = len(scored_transcript.split())

    t0 = time.time()
    per_window: list[dict[str, Any]] = []
    errors: list[str] = []

    # Sample 2..N with temperature so the spread is meaningful; by default the
    # first pass stays greedy so the headline score is reproducible. Set
    # greedy_first: false to sample every pass (needed to measure decoding
    # sensitivity, since repeated greedy runs are identical by construction).
    temps = [temperature if (s_idx > 0 or not greedy_first) else 0.0 for s_idx in range(samples)]
    total_passes = len(windows) * samples
    passes_done = 0
    # On Apple Silicon each chunk is read once and reused for all its passes
    # (identical answers, ~2x faster); elsewhere passes run one at a time.
    reuse = load is not None and sys.platform == "darwin" and stream_generate is not None

    def _tick(w_idx: int, s_idx: int) -> None:
        nonlocal passes_done
        passes_done += 1
        if progress is not None:
            progress(passes_done / total_passes,
                     f"chunk {w_idx + 1} of {len(windows)}, run {s_idx + 1} of {samples}")

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise ScoringCancelled("Scoring stopped.")

    for w_idx, window in enumerate(windows):
        prompt = CLINICAL_SCORING_PROMPT.format(
            transcript=window,
            acoustic_context=acoustic_context,
        )
        raws: list[Optional[str]] = []
        if reuse:
            try:
                raws = call_local_lm_reusing_prompt(
                    prompt, temps, model, max_tokens=max_tokens,
                    should_cancel=should_cancel,
                    on_pass=lambda s_idx, w=w_idx: _tick(w, s_idx),
                )
            except ScoringCancelled:
                raise
            except Exception as exc:
                log.error("LLM generation failed (window %d): %s", w_idx + 1, exc)
                errors.append(str(exc))
                raws = []
        else:
            for s_idx, temp in enumerate(temps):
                _check_cancel()
                try:
                    raws.append(call_local_lm(
                        prompt=prompt, model_name=model, timeout=timeout,
                        hf_model_name=hf_model, max_tokens=max_tokens, temperature=temp,
                    ))
                except Exception as exc:
                    log.error("LLM generation failed (window %d): %s", w_idx + 1, exc)
                    errors.append(str(exc))
                    raws.append(None)
                _tick(w_idx, s_idx)

        for raw in raws:
            if raw is None:
                continue
            try:
                parsed = parse_scoring_response(raw)
            except ValueError as exc:
                log.error("Failed to parse LLM response (window %d): %s", w_idx + 1, exc)
                errors.append(str(exc))
                continue
            parsed["_window"] = w_idx
            per_window.append(parsed)

    elapsed = round(time.time() - t0, 1)

    if not per_window:
        result = dict(DEFAULT_SCORES)
        result["_meta"] = {
            "model": model,
            "elapsed_seconds": elapsed,
            "error": errors[0] if errors else "no usable LLM response",
            "windows": len(windows),
            "transcript_words": total_words,
        }
        return result

    scoring = _aggregate_scores(per_window)
    scoring["_meta"] = {
        "model": model,
        "elapsed_seconds": elapsed,
        "windows": len(windows),
        "samples_per_window": samples,
        "runs_used": len(per_window),
        # Passes whose answer could not be used (generation error or malformed
        # JSON). The reliability shown is for the runs actually averaged.
        "runs_expected": total_passes,
        "runs_failed": total_passes - len(per_window),
        "transcript_words": total_words,
        "window_words": window_words,
        "transcript_scope": scope,
        # Windowing means the whole transcript is scored; this stays 1.0 unless
        # a window failed outright.
        "coverage": round(len({r["_window"] for r in per_window}) / len(windows), 3),
        "temperature": temperature,
    }
    if errors:
        scoring["_meta"]["errors"] = errors[:5]
    if len(windows) > 1:
        scoring["_meta"]["per_window_scores"] = [
            {k: r.get(k) for k in REQUIRED_SCORE_KEYS} | {"window": r["_window"]}
            for r in per_window
        ]

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
