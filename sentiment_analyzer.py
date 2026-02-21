#!/usr/bin/env python3
"""
Sentiment Analysis Module for ClinicalWhisper
Analyzes transcript text to extract sentiment insights:
  - Overall sentiment score (1-10)
  - Segment-by-segment breakdown
  - Emotional tone indicators
  - Critical moments (significant sentiment shifts)

Uses TextBlob for local NLP — no data leaves the machine.
"""

import re
from textblob import TextBlob


# ---------------------------------------------------------------------------
# Tone detection keywords
# ---------------------------------------------------------------------------
TONE_KEYWORDS = {
    "collaborative": [
        "we", "together", "let's", "team", "us", "help", "share",
        "cooperate", "agree", "work with",
    ],
    "supportive": [
        "good job", "great", "well done", "nice", "perfect", "kudos",
        "encourage", "proud", "excellent", "love it", "awesome",
    ],
    "enthusiastic": [
        "excited", "amazing", "fantastic", "wow", "incredible", "love",
        "thrilled", "can't wait", "brilliant",
    ],
    "constructive": [
        "improve", "suggest", "consider", "feedback", "remember",
        "keep in mind", "practice", "next time", "work on",
    ],
    "tense": [
        "disagree", "concern", "issue", "problem", "wrong", "mistake",
        "confus", "frustrated", "upset", "difficult", "struggle",
    ],
    "confused": [
        "don't understand", "not sure", "confused", "what do you mean",
        "unclear", "lost", "huh", "what?",
    ],
    "neutral": [
        "okay", "alright", "sure", "fine", "yeah", "yes", "no",
    ],
}


def _split_into_segments(text: str, sentences_per_segment: int = 5) -> list[str]:
    """Split text into segments of roughly N sentences each."""
    # Use regex to split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short fragments
    sentences = [s for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return [text]

    segments = []
    for i in range(0, len(sentences), sentences_per_segment):
        chunk = " ".join(sentences[i : i + sentences_per_segment])
        if chunk.strip():
            segments.append(chunk)
    return segments or [text]


def _adjust_clinical_polarity(text: str, base_polarity: float) -> float:
    """Adjust TextBlob polarity by detecting clinical negative phrases that simple NLP misses."""
    lower_text = text.lower()
    penalty = 0.0
    
    # Phrases that TextBlob gets wrong (e.g. negated positive words)
    negative_phrases = [
        "can't seem to enjoy", "can't enjoy", "don't enjoy", "exhausted",
        "can't get out of bed", "feels meh", "anhedonia", "can't get any sleep",
        "depressed", "hopeless", "not feeling great"
    ]
    
    for phrase in negative_phrases:
        if phrase in lower_text:
            penalty += 0.4  # strong clinical negative indicator
            
    return max(-1.0, base_polarity - penalty)


def _polarity_to_score(polarity: float) -> int:
    """Map TextBlob polarity (-1 … +1) to a 1–10 score."""
    # Linear mapping: -1 → 1, 0 → 5.5, +1 → 10
    return max(1, min(10, round(polarity * 4.5 + 5.5)))


def _label_from_polarity(polarity: float) -> str:
    if polarity > 0.45:
        return "Positive"
    elif polarity < -0.10:
        return "Negative"
    elif -0.10 <= polarity <= 0.45:
        return "Neutral"
    else:
        return "Mixed"


def _detect_tones(text: str) -> list[str]:
    """Detect emotional tones present in the transcript."""
    text_lower = text.lower()
    tone_scores: dict[str, int] = {}

    for tone, keywords in TONE_KEYWORDS.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        if count >= 2:  # require at least 2 hits
            tone_scores[tone] = count

    if not tone_scores:
        return ["neutral"]

    # Return tones sorted by frequency, top 4
    sorted_tones = sorted(tone_scores, key=tone_scores.get, reverse=True)
    return sorted_tones[:4]


def _find_critical_moments(segments: list[dict]) -> list[dict]:
    """Identify segments where sentiment shifted significantly."""
    moments = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]["polarity"]
        curr = segments[i]["polarity"]
        shift = curr - prev

        if abs(shift) >= 0.25:
            direction = "improved" if shift > 0 else "declined"
            moments.append({
                "segment_index": i + 1,
                "direction": direction,
                "shift": round(shift, 3),
                "from_score": segments[i - 1]["score"],
                "to_score": segments[i]["score"],
                "preview": segments[i]["preview"],
            })
    return moments


def analyze_sentiment(transcript_text: str) -> dict:
    """
    Analyze the sentiment of a transcript.

    Returns:
        {
            "overall": {"label", "score", "polarity", "subjectivity"},
            "tones": [...],
            "segments": [{"index", "score", "label", "polarity", "preview"}, ...],
            "critical_moments": [...],
            "speaker_sentiments": {}   # populated by analyze_per_speaker()
        }
    """
    if not transcript_text or not transcript_text.strip():
        return {
            "overall": {"label": "Neutral", "score": 5, "polarity": 0.0, "subjectivity": 0.0},
            "tones": ["neutral"],
            "segments": [],
            "critical_moments": [],
            "speaker_sentiments": {},
        }

    # --- Overall analysis ---
    blob = TextBlob(transcript_text)
    overall_polarity = _adjust_clinical_polarity(transcript_text, blob.sentiment.polarity)
    overall_subjectivity = blob.sentiment.subjectivity

    # --- Segment analysis ---
    raw_segments = _split_into_segments(transcript_text)
    segments = []
    for idx, seg_text in enumerate(raw_segments):
        seg_blob = TextBlob(seg_text)
        pol = _adjust_clinical_polarity(seg_text, seg_blob.sentiment.polarity)
        segments.append({
            "index": idx + 1,
            "score": _polarity_to_score(pol),
            "label": _label_from_polarity(pol),
            "polarity": round(pol, 3),
            "preview": seg_text[:120].replace("\n", " ") + ("…" if len(seg_text) > 120 else ""),
        })

    # --- Tones ---
    tones = _detect_tones(transcript_text)

    # --- Critical moments ---
    critical_moments = _find_critical_moments(segments)

    return {
        "overall": {
            "label": _label_from_polarity(overall_polarity),
            "score": _polarity_to_score(overall_polarity),
            "polarity": round(overall_polarity, 3),
            "subjectivity": round(overall_subjectivity, 3),
        },
        "tones": tones,
        "segments": segments,
        "critical_moments": critical_moments,
        "speaker_sentiments": {},
    }


def analyze_per_speaker(speaker_texts: dict[str, str]) -> dict[str, dict]:
    """
    Analyze sentiment for each speaker individually.

    Args:
        speaker_texts: {"Speaker 1": "all their text...", "Speaker 2": "..."}

    Returns:
        {
            "Speaker 1": {"label", "score", "polarity", "subjectivity", "tones", "word_count"},
            "Speaker 2": {...},
        }
    """
    results = {}
    for speaker, text in speaker_texts.items():
        if not text.strip():
            continue
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity
        tones = _detect_tones(text)
        results[speaker] = {
            "label": _label_from_polarity(pol),
            "score": _polarity_to_score(pol),
            "polarity": round(pol, 3),
            "subjectivity": round(sub, 3),
            "tones": tones,
            "word_count": len(text.split()),
        }
    return results


# ---------------------------------------------------------------------------
# Markdown formatting helpers
# ---------------------------------------------------------------------------

def _score_bar(score: int, max_score: int = 10) -> str:
    """Create a visual score bar like: ████████░░ 8/10"""
    filled = "█" * score
    empty = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def _sentiment_emoji(label: str) -> str:
    return {
        "Positive": "😊",
        "Negative": "😟",
        "Neutral": "😐",
        "Mixed": "🤔",
    }.get(label, "❓")


def _format_speaker_section(speaker_sentiments: dict) -> str:
    """Format per-speaker sentiment as Markdown."""
    if not speaker_sentiments:
        return ""

    lines = []
    lines.append("### 👥 Per-Speaker Sentiment\n")
    lines.append("| Speaker | Sentiment | Score | Tones | Words |")
    lines.append("|---------|-----------|-------|-------|-------|")

    for spk, data in sorted(speaker_sentiments.items()):
        emoji = _sentiment_emoji(data["label"])
        tones = ", ".join(data["tones"][:3])
        lines.append(
            f"| {spk} | {emoji} {data['label']} | {data['score']}/10 | {tones} | {data['word_count']:,} |"
        )

    lines.append("")

    # Add detail blocks for each speaker
    for spk, data in sorted(speaker_sentiments.items()):
        emoji = _sentiment_emoji(data["label"])
        lines.append(f"**{spk}** {emoji}  ")
        lines.append(f"Score: `{_score_bar(data['score'])}` · Subjectivity: {data['subjectivity']:.2f}  ")
        tone_tags = " · ".join(f"`{t}`" for t in data["tones"])
        lines.append(f"Tones: {tone_tags}\n")

    return "\n".join(lines)


def format_sentiment_markdown(analysis: dict) -> str:
    """Format sentiment analysis results as a Markdown section."""
    overall = analysis["overall"]
    tones = analysis["tones"]
    segments = analysis["segments"]
    moments = analysis["critical_moments"]
    speaker_sentiments = analysis.get("speaker_sentiments", {})

    lines = []
    lines.append("\n---\n")
    lines.append("## 📊 Sentiment Analysis\n")

    # Overall
    emoji = _sentiment_emoji(overall["label"])
    lines.append(f"**Overall Sentiment:** {emoji} {overall['label']}  ")
    lines.append(f"**Score:** `{_score_bar(overall['score'])}`  ")
    lines.append(f"**Subjectivity:** {overall['subjectivity']:.2f} / 1.00  \n")

    # Tones
    tone_tags = " · ".join(f"`{t}`" for t in tones)
    lines.append(f"**🎭 Emotional Tones:** {tone_tags}\n")

    # Per-speaker breakdown (if available)
    if speaker_sentiments:
        lines.append(_format_speaker_section(speaker_sentiments))

    # Segment breakdown
    if segments:
        lines.append("### Segment-by-Segment Breakdown\n")
        lines.append("| # | Score | Sentiment | Preview |")
        lines.append("|---|-------|-----------|---------|")
        for seg in segments:
            preview = seg["preview"].replace("|", "\\|")
            lines.append(
                f"| {seg['index']} | {seg['score']}/10 | {seg['label']} | {preview} |"
            )
        lines.append("")

    # Critical moments
    if moments:
        lines.append("### ⚡ Critical Moments\n")
        for m in moments:
            icon = "📈" if m["direction"] == "improved" else "📉"
            lines.append(
                f"- {icon} **Segment {m['segment_index']}** — "
                f"Sentiment {m['direction']} from {m['from_score']}/10 → {m['to_score']}/10  "
            )
            lines.append(f"  > _{m['preview']}_\n")
    else:
        lines.append("_No significant sentiment shifts detected._\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Great job on that assignment! You did really well. "
        "I'm so proud of your progress. Let's keep working together. "
        "Okay, now let's look at the next problem. I see you're confused. "
        "What don't you understand? Tell me. That doesn't make sense to me either. "
        "Actually wait, I think you misunderstood the question. "
        "No, that's wrong. Let me explain again. This is frustrating. "
        "Okay let's try a different approach. Does that help? "
        "Yes! Perfect! You got it! Excellent work! I love it. "
        "Amazing improvement from earlier. You should be proud of yourself. "
        "Alright, we're done for today. See you tomorrow. Bye!"
    )

    print("=" * 60)
    print("SENTIMENT ANALYSIS — TEST RUN")
    print("=" * 60)

    result = analyze_sentiment(sample)

    # Simulate per-speaker data
    speaker_texts = {
        "Speaker 1": "Great job on that assignment! You did really well. I'm so proud of your progress. "
                     "Let's keep working together. No, that's wrong. Let me explain again.",
        "Speaker 2": "I see you're confused. What don't you understand? "
                     "Yes! Perfect! You got it! Alright, we're done for today.",
    }
    result["speaker_sentiments"] = analyze_per_speaker(speaker_texts)

    print(f"\nOverall: {result['overall']}")
    print(f"Tones:   {result['tones']}")
    print(f"Segments: {len(result['segments'])}")
    for seg in result["segments"]:
        print(f"  [{seg['index']}] {seg['label']} ({seg['score']}/10) — {seg['preview']}")
    print(f"Critical moments: {len(result['critical_moments'])}")
    print(f"\nPer-speaker:")
    for spk, data in result["speaker_sentiments"].items():
        print(f"  {spk}: {data['label']} ({data['score']}/10) — tones: {data['tones']}")

    print("\n--- MARKDOWN OUTPUT ---")
    print(format_sentiment_markdown(result))

