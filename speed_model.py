"""How long processing will take on this Mac, learned from its own runs.

Two rates, in seconds of processing per second of audio: transcription
(including name masking and voice features) and clinical scoring. Each
finished run updates them (a moving average), stored beside the results, so
the estimate reflects this machine rather than the developer's. Before the
first run, defaults measured on an M2 Max are used, scaled by memory as a
rough stand-in for a slower chip; estimates say when they are rough.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Measured on an M2 Max (36 GB): a 3 h 25 min interview took 11.5 min to
# transcribe, mask and extract voice features, and 16.5 min to score.
_FAST = {"transcribe": 0.056, "score": 0.080}
# Machines with less memory usually have fewer GPU cores and less memory
# bandwidth; assume about four times slower until measured.
_SLOW_FACTOR = 4.0
_MIN_AUDIO_S = 120.0  # shorter files are dominated by model loading
_ALPHA = 0.3


def _path() -> Path:
    from cw_config import DATA_ROOT
    return DATA_ROOT / "speed.json"


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 32.0


def load() -> dict:
    """{"transcribe": rate, "score": rate, "runs": n}, measured or default."""
    try:
        data = json.loads(_path().read_text())
        if data.get("transcribe"):
            return data
    except (OSError, ValueError):
        pass
    factor = 1.0 if _ram_gb() >= 30 else _SLOW_FACTOR
    return {k: v * factor for k, v in _FAST.items()} | {"runs": 0}


def record(kind: str, audio_s: float | None, elapsed_s: float) -> None:
    """Fold one measured run into the stored rate for ``kind``."""
    if not audio_s or audio_s < _MIN_AUDIO_S or elapsed_s <= 0:
        return
    data = load()
    rate = elapsed_s / audio_s
    # The first measurement of each kind replaces the default outright.
    seen = data.get(f"{kind}_runs", 0)
    data[kind] = rate if not seen else (1 - _ALPHA) * data[kind] + _ALPHA * rate
    data[f"{kind}_runs"] = seen + 1
    data["runs"] = data.get("runs", 0) + 1
    try:
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(data))
    except OSError:
        pass


def estimate(audio_s: float, scoring: bool) -> tuple[float, bool]:
    """(seconds, rough). Rough until this Mac has finished a few runs."""
    data = load()
    seconds = audio_s * data["transcribe"] + (audio_s * data["score"] if scoring else 0.0)
    return seconds, data.get("runs", 0) < 3


def scored(analysis_path) -> bool:
    """Whether a finished analysis actually ran clinical scoring.

    Scoring is skipped for group recordings, other languages and unconfirmed
    roles; timing those runs as "scoring" would record a near-zero rate and
    make every later estimate far too short.
    """
    import json

    try:
        data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(data.get("llm_clinical_scoring"))
