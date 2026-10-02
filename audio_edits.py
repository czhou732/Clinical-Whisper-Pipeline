"""Leave parts of a recording out before it is processed.

A recording often holds more than the interview: setup chatter before it,
small talk after it, a stretch where the participant asked to stop, or audio a
consent form excludes. The user marks where to start, where to end, and any
stretches to skip; only the rest is decoded, transcribed and measured.

Timestamps stay those of the original recording. The kept stretches are joined
with a short silence between them (so words on either side of a cut are not run
together) and every segment time is mapped back afterwards, so "12:30" in the
transcript is still 12:30 in the file the user has.

Timing measures (pauses, response latency) are taken before mapping back: on
the original timeline a skipped hour would look like an hour-long pause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

# Silence placed between kept stretches.
GAP_S = 0.5
_TIME = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)\s*$")


def parse_time(text) -> Optional[float]:
    """Seconds from "45", "12:30", "1:02:03" or a number; None when blank."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip()
    if not text:
        return None
    m = _TIME.match(text)
    if not m:
        raise ValueError(f"Not a time: {text!r} (use mm:ss or h:mm:ss)")
    a, b, c = m.groups()
    parts = [float(x) for x in (a, b, c) if x is not None]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """"02:30-05:00, 1:10:00-1:12:00" -> [(150, 300), (4200, 4320)]."""
    out = []
    for chunk in re.split(r"[;,\n]+", text or ""):
        if not chunk.strip():
            continue
        parts = re.split(r"\s*(?:-|–|—|to)\s*", chunk.strip())
        if len(parts) != 2:
            raise ValueError(f"Not a range: {chunk.strip()!r} (use start-end, e.g. 02:30-05:00)")
        out.append((parse_time(parts[0]), parse_time(parts[1])))
    return out


@dataclass(frozen=True)
class Edits:
    start: float = 0.0
    end: Optional[float] = None
    skip: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Edits"]:
        """From the window or an IDs CSV; None when nothing is edited."""
        if not d:
            return None
        start = parse_time(d.get("start")) or 0.0
        end = parse_time(d.get("end"))
        skip = d.get("skip") or []
        if isinstance(skip, str):
            skip = parse_ranges(skip)
        edits = cls(start, end, tuple((float(a), float(b)) for a, b in skip))
        edits.validate()
        return None if edits.is_empty() else edits

    def is_empty(self) -> bool:
        return self.start <= 0 and self.end is None and not self.skip

    def validate(self) -> None:
        if self.start < 0:
            raise ValueError("Start can't be negative.")
        if self.end is not None and self.end <= self.start:
            raise ValueError("End must come after start.")
        for a, b in self.skip:
            if b <= a:
                raise ValueError(f"Skipped range {fmt(a)}-{fmt(b)} ends before it starts.")

    def keep(self) -> list[tuple[float, Optional[float]]]:
        """Kept stretches in order; the last may run to the end (None)."""
        cuts = sorted((a, b) for a, b in self.skip if self.end is None or a < self.end)
        out: list[tuple[float, Optional[float]]] = []
        pos = self.start
        for a, b in cuts:
            if b <= pos:
                continue
            if a > pos:
                out.append((pos, a))
            pos = max(pos, b)
        if self.end is None or pos < self.end:
            out.append((pos, self.end))
        return out

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "skip": [list(r) for r in self.skip]}

    def describe(self) -> str:
        parts = []
        if self.start > 0:
            parts.append(f"started at {fmt(self.start)}")
        if self.end is not None:
            parts.append(f"ended at {fmt(self.end)}")
        if self.skip:
            parts.append("skipped " + ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in self.skip))
        return "Edited before processing: " + "; ".join(parts) + "."


def fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60:02d}:{rest % 60:02d}"


def apply(blocks: Iterable[np.ndarray], edits: Edits, sr: int) -> Iterator[np.ndarray]:
    """Pass through only the kept stretches of a stream of audio blocks."""
    keep = edits.keep()
    bounds = [(int(round(a * sr)), None if b is None else int(round(b * sr))) for a, b in keep]
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    pos = 0          # sample index in the original stream
    current = 0      # index into bounds
    emitted_any = False
    started = False  # has the current stretch emitted anything yet
    for block in blocks:
        n = block.size
        lo = 0
        while lo < n and current < len(bounds):
            a, b = bounds[current]
            abs_lo = pos + lo
            if abs_lo < a:  # before this stretch: skip ahead
                lo = min(n, a - pos)
                continue
            hi = n if b is None else max(lo, min(n, b - pos))
            if hi > lo:
                if not started and emitted_any:
                    yield gap
                started = True
                emitted_any = True
                yield block[lo:hi]
            lo = hi
            if b is not None and pos + lo >= b:
                current += 1
                started = False
        pos += n
        if current >= len(bounds):
            break


@dataclass(frozen=True)
class TimeMap:
    """Maps times in the edited audio back to the original recording."""

    keep: tuple[tuple[float, Optional[float]], ...]
    gap: float = GAP_S

    @classmethod
    def for_edits(cls, edits: Optional[Edits]) -> Optional["TimeMap"]:
        return cls(tuple(edits.keep())) if edits else None

    def to_original(self, t: float, after_cut: bool = False) -> float:
        """Edited time -> original time.

        A time inside an inserted silence maps to the cut: to where the
        recording resumes when ``after_cut`` (a segment's start), else to
        where it was cut (a segment's end).
        """
        offset = 0.0  # where the current stretch starts in the edited audio
        for i, (a, b) in enumerate(self.keep):
            length = float("inf") if b is None else b - a
            if t <= offset + length:
                return a + max(0.0, t - offset)
            offset += length
            if t < offset + self.gap and i + 1 < len(self.keep):
                return float(self.keep[i + 1][0] if after_cut else b)
            offset += self.gap
        a, b = self.keep[-1]
        return float(b if b is not None else a)

    def to_edited(self, t: float) -> float:
        """Original time -> time in the edited audio (clamped into kept audio)."""
        offset = 0.0
        for i, (a, b) in enumerate(self.keep):
            if t < a:
                return offset  # inside a skipped stretch: the next kept point
            if b is None or t <= b:
                return offset + (t - a)
            offset += (b - a) + (self.gap if i + 1 < len(self.keep) else 0.0)
        return offset

    def edited_segments(self, segments: list[dict]) -> list[dict]:
        out = []
        for seg in segments:
            seg = dict(seg)
            seg["start"] = round(self.to_edited(float(seg.get("start", 0.0))), 3)
            seg["end"] = round(self.to_edited(float(seg.get("end", 0.0))), 3)
            out.append(seg)
        return out

    def segments(self, segments: list[dict]) -> list[dict]:
        out = []
        for seg in segments:
            seg = dict(seg)
            seg["start"] = round(self.to_original(float(seg.get("start", 0.0)), True), 3)
            seg["end"] = round(self.to_original(float(seg.get("end", 0.0))), 3)
            out.append(seg)
        return out
