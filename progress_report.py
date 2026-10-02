"""Progress and time remaining for the batch command.

A 220-file run lasts days; without this a working run and a stuck one look the
same. Lines are logged at stage changes and at most every ``interval`` seconds.
Time left in a stage comes from its elapsed time and reported fraction; time
left in the batch from the average processing time per second of audio so far,
which already includes transcription, masking, voice features and scoring.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("ClinicalWhisper.batch")


def _fmt(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"about {minutes} min"
    return f"about {minutes // 60} h {minutes % 60:02d} min"


def audio_seconds(path) -> Optional[float]:
    """Duration without decoding, or None if it cannot be read."""
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return container.duration / 1_000_000
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception:
        return None


class BatchProgress:
    def __init__(self, total_files: int, total_audio_s: float, interval: float = 30.0,
                 clock=time.monotonic):
        self.total_files = total_files
        self.total_audio_s = max(total_audio_s, 0.0)
        self.files_done = 0
        self.audio_done_s = 0.0
        self.interval = interval
        self._clock = clock
        self._t0 = clock()
        self._stage: Optional[str] = None
        self._stage_t0 = self._t0
        self._last = -1e9

    def batch_remaining(self) -> Optional[float]:
        if self.audio_done_s <= 0 or self.total_audio_s <= 0:
            return None
        per_audio_s = (self._clock() - self._t0) / self.audio_done_s
        return per_audio_s * max(self.total_audio_s - self.audio_done_s, 0.0)

    def update(self, stage: str, fraction: Optional[float] = None, detail: str = "") -> str:
        """Progress callback for the pipeline (``progress_cb``). Returns the line."""
        now = self._clock()
        changed = stage != self._stage
        if changed:
            self._stage, self._stage_t0 = stage, now
        if not changed and now - self._last < self.interval:
            return ""
        self._last = now
        parts = [f"{self.files_done}/{self.total_files} files done", stage]
        if fraction is not None and 0 < fraction < 1:
            parts[-1] += f" {fraction:.0%}" + (f" ({detail})" if detail else "")
            if fraction > 0.03:
                left = (now - self._stage_t0) * (1 - fraction) / fraction
                parts.append(f"{_fmt(left)} left in this stage")
        elif detail:
            parts[-1] += f" ({detail})"
        batch_left = self.batch_remaining()
        if batch_left is not None:
            parts.append(f"{_fmt(batch_left)} left in the batch")
        line = " · ".join(parts)
        log.info("Progress: %s", line)
        return line

    def file_done(self, audio_s: Optional[float]) -> None:
        self.files_done += 1
        self.audio_done_s += audio_s or 0.0
