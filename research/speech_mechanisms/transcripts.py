"""Load diarized interviews into one format, from either source.

Every loader returns segments ``[{"role", "start", "end", "text"}]`` where
``role`` is ``"interviewer"`` or ``"participant"``, sorted by start time.

* **DAIC-WOZ human transcripts** (``<id>_TRANSCRIPT.csv``, tab-separated
  ``start_time stop_time speaker value``; speakers ``Ellie`` / ``Participant``).
  These are the human-verified timing reference.
* **ClinicalWhisper output** (``[{"speaker", "start", "end", "text"}]`` JSON).
  Speaker labels are anonymous, so the interviewer is inferred as the speaker
  who asks the most questions, and the participant as the non-interviewer
  with the most talk time; any other speakers are dropped.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

INTERVIEWER, PARTICIPANT = "interviewer", "participant"


def load_daic(path: Path) -> list[dict]:
    segments = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            speaker = (row.get("speaker") or "").strip().lower()
            if speaker not in ("ellie", "participant"):
                continue
            segments.append({
                "role": INTERVIEWER if speaker == "ellie" else PARTICIPANT,
                "start": float(row["start_time"]),
                "end": float(row["stop_time"]),
                "text": (row.get("value") or "").strip(),
            })
    return sorted(segments, key=lambda s: s["start"])


def _is_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(
        re.match(r"^(what|when|where|who|why|how|do|did|are|is|can|could|would|tell me)\b",
                 text.strip().lower())
    )


def assign_roles(segments: list[dict]) -> list[dict]:
    """Label anonymous speakers: interviewer asks the most questions."""
    questions = Counter(s["speaker"] for s in segments if _is_question(s.get("text", "")))
    if not questions:
        raise ValueError("no questions found; cannot tell the interviewer apart")
    interviewer = questions.most_common(1)[0][0]
    talk: Counter = Counter()
    for s in segments:
        if s["speaker"] != interviewer:
            talk[s["speaker"]] += s["end"] - s["start"]
    if not talk:
        raise ValueError("only one speaker found")
    participant = talk.most_common(1)[0][0]
    roles = {interviewer: INTERVIEWER, participant: PARTICIPANT}
    return sorted(
        ({"role": roles[s["speaker"]], "start": s["start"], "end": s["end"], "text": s["text"]}
         for s in segments if s["speaker"] in roles),
        key=lambda s: s["start"],
    )


def load_clinicalwhisper(path: Path) -> list[dict]:
    return assign_roles(json.loads(Path(path).read_text()))


def load_dir(directory: Path, source: str) -> dict[str, list[dict]]:
    """``{participant_id: segments}`` for every transcript in ``directory``."""
    directory = Path(directory)
    if source == "daic":
        return {p.name.split("_")[0]: load_daic(p) for p in sorted(directory.glob("*_TRANSCRIPT.csv"))}
    if source == "cw":
        return {p.stem.split(".")[0]: load_clinicalwhisper(p) for p in sorted(directory.glob("*.json"))}
    raise ValueError(f"unknown source {source!r}; use 'daic' or 'cw'")


def prompt_inventory(transcripts: dict[str, list[dict]]) -> list[tuple[str, int]]:
    """Every distinct interviewer utterance and how many interviews contain it.

    DAIC-WOZ's interviewer draws on a fixed set of prompts; this list is what
    raters label for valence (blind to outcomes) before any analysis runs.
    """
    seen: Counter = Counter()
    for segs in transcripts.values():
        texts = {normalize_prompt(s["text"]) for s in segs if s["role"] == INTERVIEWER}
        seen.update(t for t in texts if t)
    return seen.most_common()


def normalize_prompt(text: str) -> str:
    text = re.sub(r"\(.*?\)|<.*?>|\[.*?\]", " ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text)).strip()
