"""Point a clinician at passages worth reading: a keyword screen, not an assessment.

A multi-hour interview is too long to reread in full, but a few things said in
it must not be missed: mentions of suicide or self-harm, harm to others, abuse,
psychotic experiences, substance use, hopelessness. This finds participant
statements that use those words and lists them with timestamps, each with the
interviewer's preceding line for context.

Deliberately simple and over-inclusive:

* Keyword matching misses concerns said in other words, and a mistranscribed
  word cannot match at all. An empty list means "no keywords", never "no risk".
* Negations are flagged too. "I've never thought about suicide" is the answer
  to a screening question, and the clinician should see that answer.
* Interviewer lines are not flagged (structured interviews ask about all of
  these), only used as context.
"""

from __future__ import annotations

import re
from typing import Optional

DISCLAIMER = ("Keyword matches for clinician review. They miss concerns said in other "
              "words or mistranscribed, and flag statements that are not concerns. "
              "This is not a risk assessment.")
NONE_FOUND = ("No keyword matches. This does not mean nothing concerning was said; "
              "the transcript still needs to be read.")

# (category, label, patterns). Patterns are matched case-insensitively against
# the masked transcript, on word boundaries.
CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("suicide_self_harm", "Suicide or self-harm", [
        r"suicid\w*", r"kill(?:ing)? myself", r"end(?:ing)? (?:my|it) (?:life|all)",
        r"take my (?:own )?life", r"want(?:ed)? to die", r"wish(?:ed)? I (?:was|were) dead",
        r"better off dead", r"(?:don'?t|do not) want to (?:live|be alive|be here|wake up)",
        r"no reason to live", r"nothing to live for", r"self[- ]harm\w*",
        r"(?:hurt|hurting|cut|cutting|burn|burning) myself", r"overdos\w*",
        r"hang(?:ing)? myself", r"not be here anymore",
    ]),
    ("harm_to_others", "Harm to others", [
        # Intent, not accidents: "hurt her nose" in a story is not a concern.
        r"homicid\w*", r"kill (?:him|her|them|someone|somebody|people|you)",
        r"(?:want(?:ed)?|going|gonna|thought about|thinking about|think about|feel like|urge)"
        r"(?: to)? (?:hurt|harm|kill|attack) (?:him|her|them|someone|somebody|people|you)",
        r"(?:shoot|stab|strangle) (?:him|her|them|someone|somebody)",
    ]),
    ("abuse_violence", "Abuse or violence", [
        r"abus(?:e|ed|ive|ing)", r"assault\w*", r"rap(?:e|ed)", r"molest\w*",
        r"(?:beat|hit|choked) me", r"domestic violence",
    ]),
    ("psychosis", "Unusual perceptions or beliefs", [
        r"hear(?:ing)? voices", r"voices (?:tell|telling|told) me",
        r"(?:seeing|see) things (?:that|other people)",
        r"(?:being|was|am) (?:followed|watched|poisoned)",
        r"(?:watching|following) me", r"paranoi\w*",
    ]),
    ("substance", "Substance use", [
        r"relaps\w*", r"heroin", r"fentanyl", r"meth(?:amphetamine)?", r"cocaine",
        r"(?:drinking|drunk) (?:every|all) (?:day|night)", r"black(?:ed)? out",
        r"withdrawal",
    ]),
    ("hopelessness", "Hopelessness", [
        r"hopeless\w*", r"no point(?: in)? (?:living|anything|going on)",
        r"can'?t go on", r"worthless", r"give up on (?:life|everything)",
    ]),
]

_COMPILED = [(cat, label, re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.IGNORECASE))
             for cat, label, pats in CATEGORIES]


def _excluded(role: Optional[str]) -> bool:
    return role in {"Interviewer", "Moderator"} or (role or "").startswith("Moderator")


def find(segments: list[dict], roles: dict[str, str],
         extra_terms: Optional[dict[str, list[str]]] = None,
         context_chars: int = 160) -> dict:
    """Passages for clinician review, in transcript order."""
    compiled = list(_COMPILED)
    for cat, terms in (extra_terms or {}).items():
        terms = [t for t in terms if isinstance(t, str) and t.strip()]
        if terms:
            label = next((lbl for c, lbl, _ in _COMPILED if c == cat), cat.replace("_", " ").title())
            compiled.append((cat, label, re.compile(
                r"\b(?:" + "|".join(re.escape(t.strip()) for t in terms) + r")\b",
                re.IGNORECASE)))

    items: list[dict] = []
    last_question: Optional[str] = None
    for seg in segments:
        spk = seg.get("speaker")
        role = roles.get(spk, spk)
        text = " ".join((seg.get("text") or "").split())
        if not text:
            continue
        if _excluded(role):
            last_question = text
            continue
        for cat, label, rx in compiled:
            m = rx.search(text)
            if not m:
                continue
            items.append({
                "category": cat,
                "label": label,
                "term": m.group(0),
                "start": round(float(seg.get("start", 0.0)), 1),
                "end": round(float(seg.get("end", 0.0)), 1),
                "speaker": spk,
                "role": role,
                "text": text,
                "context": (last_question[-context_chars:] if last_question else None),
            })
    counts: dict[str, int] = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    return {
        "items": items,
        "counts": counts,
        "note": DISCLAIMER if items else NONE_FOUND,
    }


def summary(result: dict) -> str:
    """One CSV cell: "suicide_self_harm:2; substance:1", or empty."""
    return "; ".join(f"{k}:{v}" for k, v in sorted((result or {}).get("counts", {}).items()))
