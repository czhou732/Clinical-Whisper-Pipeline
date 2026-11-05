"""Elaboration, measured: how much the participant says after each kind of question.

The language model used to rate "elaboration on positive topics" 0-10, which
it did with an ICC of .60 per run. Elaboration is countable: label each
interviewer question by the valence of what it asks about, then count the
participant's words until the interviewer asks again. The count is the same on
every run, and the positive-minus-neutral contrast is the anhedonia hypothesis
in the pre-registration (less elaboration when asked about pleasant things).

Question valence comes from a short, inspectable word list, not a model, so
it is reproducible and can be checked by eye. A question with words from both
lists, or neither, is neutral. English only.
"""

from __future__ import annotations

import re
from statistics import mean, median
from typing import Optional

_POSITIVE = re.compile(r"\b(?:enjoy\w*|fun|happ(?:y|ier|iest|iness)|proud|best|favou?rite|"
                       r"look(?:ing)? forward|excit\w*|love|like (?:to|doing)|pleas\w*|good time|"
                       r"hobb(?:y|ies)|celebrat\w*|grateful|laugh\w*|relax\w*|passion\w*|"
                       r"accomplish\w*|achiev\w*|interest(?:s|ed)? in)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:sad\w*|depress\w*|worr\w*|anx\w*|stress\w*|upset\w*|angry|anger|"
                       r"problem\w*|difficult\w*|hard(?:est| time)?|struggl\w*|bad|worst|"
                       r"regret\w*|guilt\w*|lonel\w*|hurt\w*|pain\w*|scar\w*|afraid|fear\w*|"
                       r"lost|loss|grie\w*|cry\w*|hopeless\w*|tired|sleep(?:ing)? (?:badly|poorly)|"
                       r"trouble\w*|conflict\w*|bother\w*|annoy\w*|frustrat\w*)\b", re.I)
_QUESTION = re.compile(r"\?|^(?:so |and |but )?(?:what|how|why|when|where|who|which|do|does|did|are"
                       r"|is|was|were|can|could|would|will|have|has|tell me)\b", re.I)
_WORD = re.compile(r"[A-Za-z']+")
_TAG = re.compile(r"\[[a-z_]+_\d+\]")

VALENCES = ("positive", "neutral", "negative")


def question_valence(text: str) -> Optional[str]:
    """"positive", "negative" or "neutral" for a question; None if not a question."""
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    asked = [s for s in sentences if _QUESTION.search(s)]
    if not asked:
        return None
    q = " ".join(asked)
    pos, neg = bool(_POSITIVE.search(q)), bool(_NEGATIVE.search(q))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _words(text: str) -> int:
    return len(_WORD.findall(_TAG.sub(" name ", text)))


def measure(segments: list[dict], roles: dict[str, str]) -> Optional[dict]:
    """Words per answer after positive, neutral and negative questions.

    Only for a one-to-one interview (an "Interviewer" and a "Subject"). An
    answer is everything the subject says between one interviewer question
    and the next interviewer turn.
    """
    interviewer = {k for k, v in roles.items() if v == "Interviewer"}
    subject = {k for k, v in roles.items() if v == "Subject"}
    if not interviewer or not subject:
        return None
    answers: dict[str, list[int]] = {v: [] for v in VALENCES}
    current: Optional[str] = None
    words = 0
    for seg in segments:
        spk = seg.get("speaker")
        text = seg.get("text") or ""
        if spk in interviewer:
            if current is not None and words:
                answers[current].append(words)
            valence = question_valence(text)
            if valence is not None:
                current, words = valence, 0
            elif current is not None:
                current, words = None, 0  # a remark, not a question: stop counting
        elif spk in subject and current is not None:
            words += _words(text)
    if current is not None and words:
        answers[current].append(words)

    out: dict = {}
    for v in VALENCES:
        n = answers[v]
        out[v] = {"answers": len(n),
                  "words_mean": round(mean(n), 1) if n else None,
                  "words_median": round(median(n), 1) if n else None}
    pos, neu = out["positive"]["words_median"], out["neutral"]["words_median"]
    # Positive relative to neutral: below 1 means shorter answers to pleasant
    # topics. Needs a few answers of each kind to mean anything.
    enough = out["positive"]["answers"] >= 3 and out["neutral"]["answers"] >= 3
    out["positive_to_neutral"] = round(pos / neu, 2) if enough and pos and neu else None
    out["note"] = ("Questions are labelled by an English word list; answers counted in words "
                   "until the interviewer speaks again.")
    return out
