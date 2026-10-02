"""Remembered staff voices, so moderators are recognised in every session.

Opt-in and staff-only: a voice is added only when someone confirms a speaker
as the interviewer or a moderator in the window and ticks "Remember this
voice". What is stored is a voice embedding (192 numbers from the speaker
model), a label and a role, in Application Support on this Mac. It never goes
into results or exports. Participants' voices are never stored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

import bundled_models

LIBRARY = bundled_models.APP_SUPPORT / "voice_library.json"


def load(path: Path = LIBRARY) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [v for v in data.get("voices", []) if v.get("embedding")]


def _save(voices: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"voices": voices}, indent=1), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


def remember(label: str, role: str, embedding: np.ndarray, path: Path = LIBRARY) -> None:
    """Add a staff voice, or refresh it by averaging with the stored one."""
    label = " ".join(str(label).split())[:40]
    if not label:
        raise ValueError("Give the speaker a label before remembering their voice.")
    if role not in {"Interviewer"} and not role.startswith("Moderator"):
        raise ValueError("Only interviewer and moderator voices can be remembered.")
    v = np.asarray(embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    voices = load(path)
    for entry in voices:
        if entry["label"] == label:
            old = np.asarray(entry["embedding"], dtype=np.float32)
            n = entry.get("sessions", 1)
            mean = (old * n + v) / (n + 1)
            entry.update(embedding=[round(float(x), 5) for x in mean / (np.linalg.norm(mean) or 1.0)],
                         sessions=n + 1, updated=time.strftime("%Y-%m-%d"))
            break
    else:
        voices.append({"label": label, "role": role, "sessions": 1,
                       "embedding": [round(float(x), 5) for x in v],
                       "updated": time.strftime("%Y-%m-%d")})
    _save(voices, path)


def forget(label: str, path: Path = LIBRARY) -> bool:
    voices = load(path)
    kept = [v for v in voices if v["label"] != label]
    if len(kept) == len(voices):
        return False
    _save(kept, path)
    return True


def labels(path: Path = LIBRARY) -> list[str]:
    return [v["label"] for v in load(path)]


def embeddings_for(segments: list[dict], wav_path: str) -> Optional[dict[str, np.ndarray]]:
    """One voice embedding per speaker, from the processing copy of the audio.

    Kept in memory only (for matching and for "Remember this voice").
    """
    try:
        from moss_windowed import _voice_embedder, _WavAudio
    except ImportError:  # pragma: no cover
        return None
    voice = _voice_embedder()
    if voice is None:
        return None
    audio = _WavAudio(str(wav_path))
    try:
        return {spk: emb for spk, (emb, _talk) in voice.speakers(segments, audio).items()}
    finally:
        audio.close()
