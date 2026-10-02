"""A rule-based safety net run after OpenMED, for identifiers it scores too low.

Measured on evals/masking (400 interview-style sentences, 600 identifiers):
OpenMED catches names well but scores a name lower when the sentence is long
and dense ("my sister Elena" at 0.61 after two other names, 0.96 alone), and
misses most workplaces and schools ("Bayview High School") and some spoken
numbers. These patterns are where speech reliably signals an identifier:

* a capitalised word after a relationship word: "my sister Elena";
* a capitalised word after a title: "Doctor Feldman", "Mrs. Patel";
* a capitalised phrase ending in an institution word: "Saint Mary's Hospital";
* seven or more spoken digits: "five five five, one two three four";
* a spoken email: "maria dot lopez at gmail dot com".

Each rule only adds masking; it never unmasks. Over-masking is measured
alongside recall in evals/masking/evaluate.py.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable

_CAP = r"[A-Z][a-zA-Z'\-]+"
_RELATIONS = (r"sister|brother|mom|mother|dad|father|friend|cousin|wife|husband|son|daughter|"
              r"roommate|boss|therapist|neighbou?r|girlfriend|boyfriend|partner|aunt|uncle|"
              r"grandma|grandpa|grandmother|grandfather|niece|nephew|fianc[eé]e?|coworker|"
              r"co-worker|manager|counselor|teacher|coach|pastor|kid|baby|ex")
# Words that can follow "my sister" capitalised without being a name.
_NOT_NAMES = {"I", "And", "But", "So", "Who", "She", "He", "They", "We", "It", "The", "A", "An",
              "Is", "Was", "Has", "Had", "Said", "Says", "Says,", "Too", "Again", "Yeah", "Okay",
              "Mom", "Dad", "Because", "When", "Then", "Also", "Still", "Just"}
_TITLE = r"(?:Doctor|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Professor|Prof\.?|Nurse|Coach|Pastor|Officer|Judge)"
_INSTITUTION = (r"School|Elementary|Middle|High School|Academy|University|College|Hospital|Clinic|"
                r"Center|Centre|Church|Temple|Mosque|Bank|Pharmacy|Dental|Medical|Health|"
                r"Logistics|Warehouse|Inc\.?|LLC|Company|Corporation|Corp\.?|Group|Institute|"
                r"Foundation|Department|Prison|Shelter")
_DIGIT = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
# OpenMED sometimes masks part of a phrase first ("Saint [city_1]'s Hospital",
# "[time_1], two [time_2]"); the rules treat those tags as part of the phrase.
_TAGW = r"\[[a-z_]+(?:_\d+)?\](?:'s)?"
_WORD = rf"(?:{_CAP}|{_TAGW})"

RULES: list[tuple[str, re.Pattern]] = [
    ("first_name", re.compile(rf"\b(?i:my)\s+(?i:{_RELATIONS}),?\s+({_CAP}(?:\s+{_CAP})?)")),
    ("last_name", re.compile(rf"\b{_TITLE}\s+({_CAP})")),
    ("organization", re.compile(rf"((?:\bthe\s+)?(?:{_WORD}\s+){{1,4}}(?:{_INSTITUTION}))\b")),
    ("organization", re.compile(rf"\b(?:work|works|worked|working|job)\s+(?:at|for)\s+(?:the\s+)?"
                                rf"((?:{_CAP})(?:\s+{_CAP}){{0,3}})")),
    ("phone_number", re.compile(rf"((?:\b{_DIGIT}|{_TAGW})(?:[\s,\-]+(?:{_DIGIT}\b|{_TAGW})){{2,}})", re.I)),
    # A word left exposed just before OpenMED's own email or username tag
    # ("lucia [user_name] [email]") is the start of the address.
    ("email", re.compile(r"(?<![\w'])((?!(?:is|it|its|at|me|my|was|to|the|email|address|"
                         r"mail|and|or|it's)\s)[a-z0-9]+(?:\s+\[(?:user_name|email|url)(?:_\d+)?\])+)",
                         re.I)),
    ("email", re.compile(rf"((?:\b[a-z0-9]+|{_TAGW})(?:\s+dot\s+(?:[a-z0-9]+|{_TAGW}))*\s+at\s+"
                         rf"(?:[a-z0-9]+|{_TAGW})\s+dot\s+(?:com|org|net|edu|gov|io))\b", re.I)),
]
_TAG = re.compile(r"\[[a-z_]+(?:_\d+)?\]")


def _key(text: str) -> str:
    """Identifier key for consistent numbering; the text itself is never kept."""
    return "text:" + hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def apply(masked: str, number: Callable[[str, str], int],
          found: list | None = None) -> tuple[str, dict[str, int]]:
    """Mask what the rules find in text OpenMED already masked.

    ``number(label, key)`` returns the tag number for an identifier (the
    scrubber's registry, so a name keeps one number across the recording).
    Returns the new text and counts per label.
    """
    counts: dict[str, int] = {}
    for label, rx in RULES:
        def _sub(m: re.Match) -> str:
            value = m.group(1)
            if label == "phone_number":
                digits = len(re.findall(rf"\b{_DIGIT}\b", value, re.I))
                tags = len(re.findall(_TAGW, value))
                # Seven spoken digits, or digits run together with OpenMED's own
                # number tags ("[time_1], two [time_2]").
                if not (digits >= 7 or (digits >= 1 and tags >= 1 and digits + tags >= 3)):
                    return m.group(0)
            if label in ("first_name", "last_name") and _TAG.search(value):
                return m.group(0)
            if label == "organization" and _TAG.fullmatch(value.strip()):
                return m.group(0)
            if label == "first_name":
                words = value.split()
                while words and words[-1] in _NOT_NAMES:
                    words.pop()
                if not words or words[0] in _NOT_NAMES:
                    return m.group(0)
                value = " ".join(words)
            # Keep "the" outside the tag ("at the [organization_1]").
            lead = ""
            if label == "organization" and value.lower().startswith("the "):
                lead, value = value[:4], value[4:]
            n = number(label, _key(value))
            counts[label] = counts.get(label, 0) + 1
            if found is not None:
                found.append((value, f"[{label}_{n}]"))
            start = m.start(1) - m.start(0)
            whole = m.group(0)
            return whole[:start] + lead + f"[{label}_{n}]" + whole[start + len(lead) + len(value):]
        masked = rx.sub(_sub, masked)
    return masked, counts


def propagate(texts: list[str], found: list[tuple[str, str]]) -> tuple[list[str], int]:
    """Mask every later mention of an identifier caught anywhere in the recording.

    ``found`` holds (surface text, tag) pairs collected while masking, kept in
    memory only. Matching is whole-word and case-sensitive, and only for
    capitalised strings of three or more letters, so "Elena" is masked
    everywhere once caught, while ordinary lowercase words are left alone.
    """
    pairs = {}
    for value, tag in found:
        value = value.strip(" ,.'")
        if len(value) >= 3 and value[:1].isupper() and not _TAG.search(value):
            pairs.setdefault(value, tag)
    if not pairs:
        return texts, 0
    rx = re.compile(r"(?<![\w\[])(" + "|".join(re.escape(v) for v in
                                              sorted(pairs, key=len, reverse=True)) + r")(?![\w\]])")
    total = 0
    out = []
    for t in texts:
        new, n = rx.subn(lambda m: pairs[m.group(1)], t)
        out.append(new)
        total += n
    return out, total
