"""Who is running the conversation, and who is being interviewed.

The first version took the two people who talked most and called the more
question-asking one "Interviewer". That breaks in a focus group, where the two
who talk most are the moderators: the participants ended up as "Other", and
the participant measures described a moderator.

This combines evidence, each kind checking the others:

* known voices: staff voices remembered on this Mac (opt-in, see
  ``voice_library.py``); the strongest signal when present;
* the script: lines from the study's interview guide, if one is given, and
  the consent and housekeeping language every moderator reads;
* conversation structure: questions, handing the floor to others
  ("what about you", "let's move on"), and who speaks first.

It also decides the mode. With three or more people who each talk a real
share of the time, the recording is a group: moderators and participants,
no single "Subject", and no clinical scores. When the evidence does not
separate the moderator(s) from everyone else, it says so rather than guess,
and the participant measures wait until someone confirms the roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# A speaker counts toward "how many people are in this" above this share of
# talk time (or a minute, whichever is more).
SUBSTANTIAL_SHARE = 0.08
SUBSTANTIAL_S = 60.0
# A known moderator voice matches at this cosine similarity (WeSpeaker; the
# windowed linker links the same person across windows at 0.5).
KNOWN_VOICE = 0.6
# Below this score gap between the weakest moderator and the strongest
# participant, the roles are reported as uncertain.
UNCERTAIN_GAP = 0.6

_QUESTION = re.compile(r"\?|^(?:so |and |but )?(?:what|how|why|when|where|who|which|do|does|did|"
                       r"are|is|was|were|can|could|would|will|have|has|tell me)\b", re.I)
_FLOOR = re.compile(
    r"\b(?:what about you|how about you|and you\b|go ahead|you (?:can|may) go|"
    r"let'?s (?:move|start|go|begin|talk)|next question|anyone else|does anyone|"
    r"(?:any|other) (?:thoughts|questions)|thank you (?:for|so much)|welcome|"
    r"we'?ll (?:start|begin|move)|i'?d like to (?:ask|hear))\b", re.I)
_SCRIPT = re.compile(
    r"\b(?:participation is (?:completely |entirely )?voluntary|you (?:may|can) (?:skip|stop|leave"
    r"|withdraw)|with your permission|information sheet|consent form|keep (?:it |what .{0,20})?"
    r"(?:private|confidential)|no right or wrong answers|we will (?:record|remove|be recording)|"
    r"gift card|before we (?:begin|start|get started))\b", re.I)
_ADDRESS = re.compile(r"\[first_name_\d+\]\s*[,?]")


@dataclass
class Evidence:
    talk_s: float = 0.0
    words: int = 0
    turns: int = 0
    sentences: int = 0
    questions: int = 0
    floor: int = 0
    script: int = 0
    address: int = 0
    guide: float = 0.0          # share of guide questions this speaker asked
    known: Optional[str] = None  # name of the remembered voice it matches
    known_sim: float = 0.0
    first_s: float = 0.0
    score: float = 0.0
    base: float = 0.0            # score without the remembered-voice bonus
    reasons: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def _guide_questions(guide_text: Optional[str]) -> list[str]:
    if not guide_text:
        return []
    out = []
    for line in guide_text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if len(line.split()) >= 5:
            out.append(line.lower())
    return out


def _guide_share(text: str, questions: list[str]) -> float:
    if not questions:
        return 0.0
    from rapidfuzz import fuzz

    low = text.lower()
    hit = sum(1 for q in questions if fuzz.partial_ratio(q, low) >= 80)
    return hit / len(questions)


def gather(segments: list[dict], guide_text: Optional[str] = None,
           voices: Optional[dict[str, np.ndarray]] = None,
           library: Optional[list[dict]] = None) -> dict[str, Evidence]:
    """Evidence per speaker label."""
    ev: dict[str, Evidence] = {}
    texts: dict[str, list[str]] = {}
    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        e = ev.setdefault(spk, Evidence(first_s=float(seg.get("start", 0.0))))
        text = " ".join((seg.get("text") or "").split())
        e.talk_s += max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        e.turns += 1
        e.words += len(text.split())
        sents = _sentences(text)
        e.sentences += len(sents)
        e.questions += sum(1 for s in sents if _QUESTION.search(s))
        e.floor += len(_FLOOR.findall(text))
        e.script += len(_SCRIPT.findall(text))
        e.address += len(_ADDRESS.findall(text))
        texts.setdefault(spk, []).append(text)

    questions = _guide_questions(guide_text)
    for spk, e in ev.items():
        e.guide = _guide_share(" ".join(texts[spk]), questions)
        if voices and library and spk in voices:
            v = voices[spk] / (np.linalg.norm(voices[spk]) or 1.0)
            for entry in library:
                ref = np.asarray(entry["embedding"], dtype=np.float32)
                sim = float(v @ (ref / (np.linalg.norm(ref) or 1.0)))
                if sim > e.known_sim:
                    e.known_sim, e.known = sim, entry.get("label")
            if e.known_sim < KNOWN_VOICE:
                e.known = None
    return ev


def _score(e: Evidence, first: bool, most_talk: float) -> float:
    """How much a speaker behaves like the person running the conversation."""
    reasons = []
    # Share of their sentences that are questions: an answerer's long turns
    # hold the odd rhetorical question, an interviewer's are mostly questions.
    q_share = e.questions / max(1, e.sentences)
    score = 3.0 * min(1.0, q_share / 0.5)
    if e.questions:
        reasons.append(f"{round(100 * q_share)}% of sentences are questions")
    # The person interviewing usually talks less than the person interviewed.
    score += 1.5 * (1.0 - e.talk_s / most_talk) if most_talk else 0.0
    if e.floor:
        score += min(1.5, 0.3 * e.floor)
        reasons.append("hands the floor to others")
    if e.address:
        score += min(1.0, 0.25 * e.address)
        reasons.append("addresses people by name")
    if e.script:
        score += min(2.0, 1.0 * e.script)
        reasons.append("reads consent or housekeeping lines")
    if e.guide:
        score += 4.0 * e.guide
        reasons.append(f"asks {round(100 * e.guide)}% of the guide questions")
    if first:
        score += 0.3
    e.base = round(score, 2)
    if e.known:
        score += 5.0
        reasons.append(f"voice matches {e.known}")
    e.score, e.reasons = round(score, 2), reasons
    return score


def assign(segments: list[dict], guide_text: Optional[str] = None,
           voices: Optional[dict[str, np.ndarray]] = None,
           library: Optional[list[dict]] = None, group: bool = False) -> dict:
    """Roles for every speaker, the mode, and how sure the assignment is.

    Returns ``{"roles", "mode", "uncertain", "why", "evidence"}``. Roles are
    "Interviewer"/"Subject"/"Other_N" for an interview, and "Moderator N"/
    "Participant N"/"Other_N" for a group. ``group`` (the user chose "Group
    discussion") uses group roles even when only two people talk much.
    """
    ev = gather(segments, guide_text, voices, library)
    if not ev:
        return {"roles": {}, "mode": "interview", "uncertain": False, "why": "", "evidence": {}}
    order = sorted(ev, key=lambda s: ev[s].first_s)
    most_talk = max(e.talk_s for e in ev.values())
    for spk in ev:
        _score(ev[spk], spk == order[0], most_talk)
    total = sum(e.talk_s for e in ev.values()) or 1.0
    # A minute of talk, or less in a short recording (15% of its talk time).
    floor = max(SUBSTANTIAL_SHARE * total, min(SUBSTANTIAL_S, 0.15 * total))
    substantial = [s for s in ev if ev[s].talk_s >= floor]
    by_talk = sorted(ev, key=lambda s: -ev[s].talk_s)
    evidence = {s: {"score": e.score, "reasons": e.reasons, "talk_s": round(e.talk_s, 1),
                    "known_voice": e.known} for s, e in ev.items()}

    if len(ev) == 1:
        return {"roles": {by_talk[0]: "Subject"}, "mode": "interview", "uncertain": False,
                "why": "", "evidence": evidence}

    if len(substantial) <= 2 and not group:
        core = by_talk[:2]
        a, b = sorted(core, key=lambda s: -ev[s].score)
        roles = {a: "Interviewer", b: "Subject"}
        for i, spk in enumerate([s for s in by_talk if s not in core], start=1):
            roles[spk] = f"Other_{i}"
        gap = ev[a].score - ev[b].score
        certain = gap >= UNCERTAIN_GAP or bool(ev[a].known) or ev[a].guide >= 0.3
        why = "" if certain else (
            f"{a} and {b} look about equally like the interviewer (scores {ev[a].score} and "
            f"{ev[b].score}).")
        return {"roles": roles, "mode": "interview", "uncertain": not certain, "why": why,
                "evidence": evidence}

    # Group: moderators are the clear high scorers. A moderator can talk
    # little (a co-facilitator's occasional question), so the bar for them is
    # lower than the one for counting someone as a participant.
    present = [s for s in ev if ev[s].talk_s >= max(0.02 * total, min(20.0, 0.05 * total))]
    ranked = sorted(present, key=lambda s: -ev[s].score)
    # Compared on the text evidence alone: a recognised voice adds its owner
    # as a moderator but must not raise the bar for everyone else.
    top = max(ev[s].base for s in present)
    moderators = [s for s in ranked if ev[s].known or ev[s].base >= max(1.5, 0.6 * top)]
    moderators = moderators[:max(1, len(present) // 2)] or ranked[:1]
    participants = [s for s in by_talk if s in substantial and s not in moderators]
    roles: dict[str, str] = {}
    for i, spk in enumerate(sorted(moderators, key=lambda s: ev[s].first_s), start=1):
        roles[spk] = f"Moderator {i}"
    for i, spk in enumerate(participants, start=1):
        roles[spk] = f"Participant {i}"
    for i, spk in enumerate([s for s in by_talk if s not in roles], start=1):
        roles[spk] = f"Other_{i}"
    weakest_mod = min(ev[s].score for s in moderators)
    strongest_part = max((ev[s].score for s in participants), default=0.0)
    if any(ev[s].known for s in moderators):
        weakest_mod = min(ev[s].base if not ev[s].known else ev[s].score for s in moderators)
    certain = (weakest_mod - strongest_part >= UNCERTAIN_GAP
               or all(ev[s].known for s in moderators))
    why = "" if certain else (
        "The moderators don't stand out clearly from the participants "
        f"(weakest moderator {weakest_mod}, strongest participant {strongest_part}).")
    return {"roles": roles, "mode": "group", "uncertain": not certain, "why": why,
            "evidence": evidence}
