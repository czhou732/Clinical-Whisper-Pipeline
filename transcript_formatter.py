#!/usr/bin/env python3
"""
Transcript Formatter Module for ClinicalWhisper

Takes diarized segments and produces a structured clinical transcript:
  - Speaker role classification (Interviewer vs Subject)
  - Structured transcript with timestamps and merged turns
  - Per-speaker speaking statistics

All processing is local — no data leaves the machine.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("ClinicalWhisper")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def _question_ratio(text: str) -> float:
    """Fraction of sentences ending with '?'."""
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", "?.").split(".") if s.strip()]
    if not sentences:
        return 0.0
    questions = sum(1 for s in sentences if s.endswith("?"))
    return questions / len(sentences)


def _word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


# ---------------------------------------------------------------------------
# 1. Speaker Role Classification
# ---------------------------------------------------------------------------

def classify_speakers(segments: list[dict]) -> dict[str, str]:
    """
    Classify each speaker as 'Interviewer', 'Subject', or 'Other_N'.

    Uses heuristics based on clinical interview patterns:
      - Interviewer: shorter turns, higher question ratio, tends to speak first
      - Subject: longer turns, more declarative statements, more total words

    Args:
        segments: List of segment dicts with keys: start, end, speaker, text.
                  The 'sentiment' key is optional.

    Returns:
        Mapping from speaker label to role, e.g.
        ``{'Speaker 1': 'Interviewer', 'Speaker 2': 'Subject'}``.
    """
    if not segments:
        log.warning("classify_speakers called with empty segments list")
        return {}

    # ── Gather per-speaker aggregates ──
    speaker_data: dict[str, dict] = {}
    speaker_order: list[str] = []

    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        text = (seg.get("text") or "").strip()

        if spk not in speaker_data:
            speaker_data[spk] = {
                "texts": [],
                "turn_count": 0,
                "total_words": 0,
                "total_duration": 0.0,
            }
            speaker_order.append(spk)

        speaker_data[spk]["texts"].append(text)
        speaker_data[spk]["turn_count"] += 1
        speaker_data[spk]["total_words"] += _word_count(text)

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        speaker_data[spk]["total_duration"] += max(0.0, end - start)

    unique_speakers = list(speaker_order)

    # ── Single speaker: label as Subject (no interviewer identifiable) ──
    if len(unique_speakers) == 1:
        log.info("Only one speaker detected — labelling as Subject")
        return {unique_speakers[0]: "Subject"}

    # ── Score each speaker on interviewer-likeness ──
    # Higher score → more likely the interviewer.
    scores: dict[str, float] = {}

    for spk, data in speaker_data.items():
        combined_text = " ".join(data["texts"])
        avg_turn_words = data["total_words"] / max(data["turn_count"], 1)
        q_ratio = _question_ratio(combined_text)

        # Interviewers ask more questions, have shorter turns, and speak
        # fewer total words relative to the subject.
        score = 0.0

        # Question ratio is the strongest signal (0–1, weight 3)
        score += q_ratio * 3.0

        # Shorter average turns → more interviewer-like (inverse, capped)
        # Normalize by assuming avg clinical turn is ~20 words.
        score += max(0.0, 1.0 - (avg_turn_words / 40.0))

        # Fewer total words → more interviewer-like
        max_words = max(d["total_words"] for d in speaker_data.values()) or 1
        score += 1.0 - (data["total_words"] / max_words)

        # Tendency to speak first gets a small bonus
        if spk == unique_speakers[0]:
            score += 0.3

        scores[spk] = score

    # ── Assign roles ──
    ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
    roles: dict[str, str] = {}
    roles[ranked[0]] = "Interviewer"

    if len(ranked) == 2:
        roles[ranked[1]] = "Subject"
    else:
        # Two-speaker core: highest score → Interviewer, lowest → Subject,
        # everyone else → Other_N.
        roles[ranked[-1]] = "Subject"
        other_idx = 1
        for spk in ranked[1:-1]:
            roles[spk] = f"Other_{other_idx}"
            other_idx += 1

    log.info("Speaker roles: %s", roles)
    return roles


# ---------------------------------------------------------------------------
# 2. Structured Transcript Formatting
# ---------------------------------------------------------------------------

def format_structured_transcript(
    segments: list[dict],
    roles: dict[str, str],
) -> str:
    """
    Format diarized segments into a clean, timestamped clinical transcript.

    Consecutive segments from the same speaker are merged into a single turn.

    Args:
        segments: List of segment dicts with keys: start, end, speaker, text.
        roles:    Mapping from speaker label to role (from ``classify_speakers``).

    Returns:
        Multi-line formatted transcript string, e.g.::

            [00:00 - 00:15] Interviewer: How have you been feeling?
            [00:15 - 00:45] Subject: I've been okay, I guess.
    """
    if not segments:
        return ""

    # ── Merge consecutive same-speaker segments ──
    merged_turns: list[dict] = []

    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        if not text:
            continue

        if merged_turns and merged_turns[-1]["speaker"] == spk:
            # Extend the current turn
            merged_turns[-1]["end"] = end
            merged_turns[-1]["texts"].append(text)
        else:
            merged_turns.append({
                "speaker": spk,
                "start": start,
                "end": end,
                "texts": [text],
            })

    # ── Format each turn ──
    lines: list[str] = []
    for turn in merged_turns:
        ts_start = _seconds_to_mmss(turn["start"])
        ts_end = _seconds_to_mmss(turn["end"])
        role = roles.get(turn["speaker"], turn["speaker"])
        combined_text = " ".join(turn["texts"])
        lines.append(f"[{ts_start} - {ts_end}] {role}: {combined_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Speaker Statistics
# ---------------------------------------------------------------------------

def compute_speaker_stats(
    segments: list[dict],
    roles: dict[str, str],
) -> dict[str, dict]:
    """
    Compute per-speaker statistics from diarized segments.

    Args:
        segments: List of segment dicts with keys: start, end, speaker, text.
        roles:    Mapping from speaker label to role.

    Returns:
        Dict keyed by speaker label, each value containing::

            {
                "role": str,
                "word_count": int,
                "turn_count": int,
                "avg_turn_length_words": float,
                "question_ratio": float,
                "speaking_time_seconds": float,
            }
    """
    if not segments:
        return {}

    stats: dict[str, dict] = {}

    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        duration = max(0.0, end - start)

        if spk not in stats:
            stats[spk] = {
                "role": roles.get(spk, spk),
                "word_count": 0,
                "turn_count": 0,
                "speaking_time_seconds": 0.0,
                "_texts": [],
            }

        wc = _word_count(text)
        stats[spk]["word_count"] += wc
        stats[spk]["turn_count"] += 1
        stats[spk]["speaking_time_seconds"] += duration
        stats[spk]["_texts"].append(text)

    # ── Derive computed fields and clean up ──
    for spk, data in stats.items():
        turns = data["turn_count"] or 1
        data["avg_turn_length_words"] = round(data["word_count"] / turns, 2)
        combined = " ".join(data.pop("_texts"))
        data["question_ratio"] = round(_question_ratio(combined), 4)
        data["speaking_time_seconds"] = round(data["speaking_time_seconds"], 3)

    return stats


# ---------------------------------------------------------------------------
# 4. Convenience pipeline
# ---------------------------------------------------------------------------

def process_segments(segments: list[dict]) -> dict:
    """
    Run the full transcript formatting pipeline.

    1. Classify speaker roles.
    2. Produce a structured transcript with timestamps.
    3. Compute per-speaker statistics.

    Args:
        segments: List of segment dicts (start, end, speaker, text, [sentiment]).

    Returns:
        {
            "roles": {speaker_label: role, ...},
            "structured_transcript": str,
            "speaker_stats": {speaker_label: stats_dict, ...},
        }
    """
    roles = classify_speakers(segments)
    transcript = format_structured_transcript(segments, roles)
    stats = compute_speaker_stats(segments, roles)

    return {
        "roles": roles,
        "structured_transcript": transcript,
        "speaker_stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Accept a JSON analysis file from CLI, or use built-in demo data ──
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1]).expanduser()
        if not json_path.exists():
            print(f"File not found: {json_path}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
        if not segments:
            print("No 'segments' key found in JSON file.", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(segments)} segments from {json_path.name}\n")
    else:
        # Demo segments simulating a clinical interview
        segments = [
            {"start": 0.0, "end": 5.2, "speaker": "Speaker 1", "text": "Good morning. How have you been feeling this past week?", "sentiment": 6},
            {"start": 5.5, "end": 18.1, "speaker": "Speaker 2", "text": "I've been okay, I guess. Nothing really exciting happening. Mostly just staying at home and trying to keep busy.", "sentiment": 5},
            {"start": 18.5, "end": 22.0, "speaker": "Speaker 1", "text": "Have you been sleeping well?", "sentiment": 6},
            {"start": 22.3, "end": 38.7, "speaker": "Speaker 2", "text": "Not really. I keep waking up around three in the morning and then I can't fall back asleep. I just lie there thinking about things.", "sentiment": 3},
            {"start": 39.0, "end": 44.5, "speaker": "Speaker 1", "text": "What kinds of things are you thinking about when that happens?", "sentiment": 5},
            {"start": 45.0, "end": 65.2, "speaker": "Speaker 2", "text": "Just everything. Work stuff, whether I'm doing enough. Sometimes I worry about my health. It feels like my mind won't shut off.", "sentiment": 3},
            {"start": 65.5, "end": 72.0, "speaker": "Speaker 1", "text": "That sounds really difficult. Are you still taking your medication as prescribed?", "sentiment": 6},
            {"start": 72.5, "end": 85.0, "speaker": "Speaker 2", "text": "Yes, I haven't missed any doses. But honestly I'm not sure if it's helping much. I still feel anxious most days.", "sentiment": 4},
            {"start": 85.5, "end": 95.0, "speaker": "Speaker 1", "text": "Okay, we may want to discuss adjusting the dosage. Have you noticed any side effects?", "sentiment": 6},
            {"start": 95.5, "end": 108.0, "speaker": "Speaker 2", "text": "A little bit of nausea in the mornings, but it usually goes away after I eat something. Otherwise nothing major.", "sentiment": 5},
        ]
        print("Using built-in demo segments (clinical interview)\n")

    # ── Run pipeline ──
    result = process_segments(segments)

    # ── Print roles ──
    print("=" * 60)
    print("SPEAKER ROLES")
    print("=" * 60)
    for spk, role in result["roles"].items():
        print(f"  {spk} → {role}")

    # ── Print structured transcript ──
    print("\n" + "=" * 60)
    print("STRUCTURED TRANSCRIPT")
    print("=" * 60)
    print(result["structured_transcript"])

    # ── Print stats ──
    print("\n" + "=" * 60)
    print("SPEAKER STATISTICS")
    print("=" * 60)
    for spk, stats in result["speaker_stats"].items():
        print(f"\n  {spk} ({stats['role']}):")
        print(f"    Words:            {stats['word_count']}")
        print(f"    Turns:            {stats['turn_count']}")
        print(f"    Avg turn length:  {stats['avg_turn_length_words']} words")
        print(f"    Question ratio:   {stats['question_ratio']:.2%}")
        print(f"    Speaking time:    {stats['speaking_time_seconds']:.1f}s")
