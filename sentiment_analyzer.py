#!/usr/bin/env python3
"""
Sentiment Analysis Module for ClinicalWhisper
Analyzes transcript text to extract sentiment insights:
  - Overall sentiment score (1-10)
  - Segment-by-segment breakdown
  - Emotional tone indicators
  - Critical moments (significant sentiment shifts)

Uses a RoBERTa transformer (cardiffnlp/twitter-roberta-base-sentiment-latest)
for local NLP — no data leaves the machine.
"""

import re
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Lazy-loaded transformer pipeline
# ---------------------------------------------------------------------------
_sentiment_pipeline = None


def _get_pipeline():
    """Lazily load and cache the RoBERTa sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
            device=-1,  # force CPU
        )
    return _sentiment_pipeline


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_text(text: str) -> dict:
    """Run transformer on text, return {negative, neutral, positive} score dict.

    Truncates input to ~512 tokens (RoBERTa limit) by keeping the first
    ~1500 characters as a safe approximation.
    """
    truncated = text[:1500]
    pipe = _get_pipeline()
    results = pipe(truncated)[0]  # list of {label, score} dicts

    scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for item in results:
        label = item["label"].lower()
        scores[label] = item["score"]
    return scores


def _polarity_from_scores(scores: dict) -> float:
    """Compute an effective polarity (-1 to +1) from transformer scores."""
    return scores["positive"] - scores["negative"]


def _polarity_to_score(polarity: float) -> int:
    """Map effective polarity (-1 ... +1) to a 1-10 integer score."""
    return max(1, min(10, round(polarity * 4.5 + 5.5)))


def _label_from_scores(scores: dict) -> str:
    """Pick the dominant sentiment label from transformer output."""
    dominant = max(scores, key=scores.get)
    if dominant == "positive":
        return "Positive"
    elif dominant == "negative":
        return "Negative"
    else:
        return "Neutral"


def _split_into_segments(text: str, sentences_per_segment: int = 5) -> list[str]:
    """Split text into segments of roughly N sentences each."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return [text]

    segments = []
    for i in range(0, len(sentences), sentences_per_segment):
        chunk = " ".join(sentences[i : i + sentences_per_segment])
        if chunk.strip():
            segments.append(chunk)
    return segments or [text]


def _detect_tones(text: str) -> list[str]:
    """Detect emotional tones present in the transcript."""
    text_lower = text.lower()
    tone_scores: dict[str, int] = {}

    for tone, keywords in TONE_KEYWORDS.items():
        count = sum(text_lower.count(kw) for kw in keywords)
        if count >= 2:
            tone_scores[tone] = count

    if not tone_scores:
        return ["neutral"]

    sorted_tones = sorted(tone_scores, key=tone_scores.get, reverse=True)
    return sorted_tones[:4]


def _find_critical_moments(segments: list[dict]) -> list[dict]:
    """Identify segments where sentiment shifted significantly."""
    moments = []
    for i in range(1, len(segments)):
        prev = segments[i - 1]["polarity"]
        curr = segments[i]["polarity"]
        shift = curr - prev

        if abs(shift) >= 0.15:
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


# ---------------------------------------------------------------------------
# Public API -- single-segment helper for inference_pipeline.py
# ---------------------------------------------------------------------------

def _get_segment_sentiment_score(text: str) -> int:
    """Return a 1-10 sentiment score for a single text segment.

    Used by inference_pipeline.py when building per-segment metadata.
    """
    if not text or not text.strip():
        return 5
    scores = _classify_text(text)
    polarity = _polarity_from_scores(scores)
    return _polarity_to_score(polarity)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    overall_scores = _classify_text(transcript_text)
    overall_polarity = _polarity_from_scores(overall_scores)
    overall_subjectivity = 1.0 - overall_scores["neutral"]

    # --- Segment analysis ---
    raw_segments = _split_into_segments(transcript_text)
    segments = []
    for idx, seg_text in enumerate(raw_segments):
        seg_scores = _classify_text(seg_text)
        pol = _polarity_from_scores(seg_scores)
        segments.append({
            "index": idx + 1,
            "score": _polarity_to_score(pol),
            "label": _label_from_scores(seg_scores),
            "polarity": round(pol, 3),
            "preview": seg_text[:120].replace("\n", " ") + ("..." if len(seg_text) > 120 else ""),
        })

    # --- Tones ---
    tones = _detect_tones(transcript_text)

    # --- Critical moments ---
    critical_moments = _find_critical_moments(segments)

    return {
        "overall": {
            "label": _label_from_scores(overall_scores),
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
        scores = _classify_text(text)
        pol = _polarity_from_scores(scores)
        sub = 1.0 - scores["neutral"]
        tones = _detect_tones(text)
        results[speaker] = {
            "label": _label_from_scores(scores),
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
    """Create a visual score bar like: xxxxxxxx.. 8/10"""
    filled = "\u2588" * score
    empty = "\u2591" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def _sentiment_emoji(label: str) -> str:
    return {
        "Positive": "\U0001f60a",
        "Negative": "\U0001f61f",
        "Neutral": "\U0001f610",
        "Mixed": "\U0001f914",
    }.get(label, "\u2753")


def _format_speaker_section(speaker_sentiments: dict) -> str:
    """Format per-speaker sentiment as Markdown."""
    if not speaker_sentiments:
        return ""

    lines = []
    lines.append("### \U0001f465 Per-Speaker Sentiment\n")
    lines.append("| Speaker | Sentiment | Score | Tones | Words |")
    lines.append("|---------|-----------|-------|-------|-------|")

    for spk, data in sorted(speaker_sentiments.items()):
        emoji = _sentiment_emoji(data["label"])
        tones = ", ".join(data["tones"][:3])
        lines.append(
            f"| {spk} | {emoji} {data['label']} | {data['score']}/10 | {tones} | {data['word_count']:,} |"
        )

    lines.append("")

    for spk, data in sorted(speaker_sentiments.items()):
        emoji = _sentiment_emoji(data["label"])
        lines.append(f"**{spk}** {emoji}  ")
        lines.append(f"Score: `{_score_bar(data['score'])}` \u00b7 Subjectivity: {data['subjectivity']:.2f}  ")
        tone_tags = " \u00b7 ".join(f"`{t}`" for t in data["tones"])
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
    lines.append("## \U0001f4ca Sentiment Analysis\n")

    emoji = _sentiment_emoji(overall["label"])
    lines.append(f"**Overall Sentiment:** {emoji} {overall['label']}  ")
    lines.append(f"**Score:** `{_score_bar(overall['score'])}`  ")
    lines.append(f"**Subjectivity:** {overall['subjectivity']:.2f} / 1.00  \n")

    tone_tags = " \u00b7 ".join(f"`{t}`" for t in tones)
    lines.append(f"**\U0001f3ad Emotional Tones:** {tone_tags}\n")

    if speaker_sentiments:
        lines.append(_format_speaker_section(speaker_sentiments))

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

    if moments:
        lines.append("### \u26a1 Critical Moments\n")
        for m in moments:
            icon = "\U0001f4c8" if m["direction"] == "improved" else "\U0001f4c9"
            lines.append(
                f"- {icon} **Segment {m['segment_index']}** \u2014 "
                f"Sentiment {m['direction']} from {m['from_score']}/10 \u2192 {m['to_score']}/10  "
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
    print("SENTIMENT ANALYSIS -- TEST RUN (RoBERTa Transformer)")
    print("=" * 60)

    result = analyze_sentiment(sample)

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
        print(f"  [{seg['index']}] {seg['label']} ({seg['score']}/10) -- {seg['preview']}")
    print(f"Critical moments: {len(result['critical_moments'])}")
    print(f"\nPer-speaker:")
    for spk, data in result["speaker_sentiments"].items():
        print(f"  {spk}: {data['label']} ({data['score']}/10) -- tones: {data['tones']}")

    print("\n--- MARKDOWN OUTPUT ---")
    print(format_sentiment_markdown(result))
