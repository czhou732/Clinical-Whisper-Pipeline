"""Silence masked names in the audio itself, not just in the transcript.

A de-identified transcript is not enough to share a recording: the names are
still spoken in it. This writes a copy of the processed audio in which every
word the text masker replaced ([first_name_1], [city_1], ...) is silent.

How the words are found:

1. Before masking, each segment's original words are kept in memory only
   (never written anywhere).
2. After masking, the original and masked words of each segment are lined up;
   the original words that became tags are the ones to silence.
3. Only segments that contain a tag are aligned to the audio, word by word,
   with a CTC model (facebook/wav2vec2-base-960h, Apache-2.0, English; an
   add-on), and each masked word is silenced with a 150 ms margin either side.

It fails closed: a segment that can't be aligned (non-English, digits only,
alignment error) is silenced whole. What it cannot do is silence a name the
text masker missed, so the same review applies as for transcripts.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

import bundled_models

log = logging.getLogger("ClinicalWhisper")

MODEL = "facebook/wav2vec2-base-960h"
ROOT = bundled_models.APP_SUPPORT / "addons" / "deid"
SAMPLE_RATE = 16000
MARGIN_S = 0.15
_TAG = re.compile(r"^\[[a-z_]+_\d+\][.,!?;:]*$|^\[REDACTED\]")


def model_path() -> Optional[str]:
    """Local folder of the aligner: the add-on, else the Hugging Face cache."""
    for hub in (ROOT / "hub",):
        base = hub / ("models--" + MODEL.replace("/", "--"))
        try:
            rev = (base / "refs" / "main").read_text().strip()
        except OSError:
            continue
        snap = base / "snapshots" / rev
        if (snap / "model.safetensors").is_file():
            return str(snap)
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(MODEL, "model.safetensors")
        return str(Path(hit).parent) if isinstance(hit, str) else None
    except ImportError:  # pragma: no cover
        return None


def available() -> bool:
    return model_path() is not None


def masked_word_indices(raw: str, masked: str) -> list[int]:
    """Indices of the words in ``raw`` that the masker replaced with tags.

    The two word lists are lined up; any stretch of original words whose
    counterpart contains a tag, or that was dropped, counts as masked.
    """
    a, b = raw.split(), masked.split()
    norm = lambda w: re.sub(r"[^\w']", "", w.lower())  # noqa: E731
    out: list[int] = []
    sm = difflib.SequenceMatcher(a=[norm(w) for w in a], b=[norm(w) for w in b], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete" or (op == "replace" and any(_TAG.match(w) for w in b[j1:j2])):
            out.extend(range(i1, i2))
    if not out and any(_TAG.match(w) for w in b):
        out = list(range(len(a)))  # tags that can't be placed: treat every word as masked
    return out


class _Aligner:
    def __init__(self, path: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank = self.vocab["<pad>"]
        self.sep = self.vocab["|"]

    def word_times(self, audio: np.ndarray, words: list[str]) -> list[Optional[tuple[float, float]]]:
        """(start, end) in seconds within ``audio`` for each word; None if it has no letters."""
        import torchaudio.functional as F

        torch = self.torch
        letters = [re.sub(r"[^A-Z']", "", w.upper()) for w in words]
        targets, owner = [], []
        for i, w in enumerate(letters):
            if not w:
                continue
            if targets:
                targets.append(self.sep)
                owner.append(-1)
            for ch in w:
                targets.append(self.vocab.get(ch, self.vocab["<unk>"]))
                owner.append(i)
        if not targets:
            return [None] * len(words)
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        frames = log_probs.shape[1]
        if frames < len(targets):
            raise ValueError("segment too short to align")
        path, _ = F.forced_align(log_probs, torch.tensor([targets], dtype=torch.int32), blank=self.blank)
        spans = F.merge_tokens(path[0], torch.zeros(path.shape[1]))
        sec = len(audio) / SAMPLE_RATE / frames
        times: list[Optional[tuple[float, float]]] = [None] * len(words)
        for k, span in enumerate(spans):
            i = owner[k] if k < len(owner) else -1
            if i < 0:
                continue
            s, e = span.start * sec, (span.end + 1) * sec
            times[i] = (min(times[i][0], s), max(times[i][1], e)) if times[i] else (s, e)
        return times


_ALIGNER: list = []


def silence_spans(raw_segments: list[dict], masked_segments: list[dict], wav_path: str,
                  use_aligner: bool = True) -> dict:
    """Time spans (seconds, in ``wav_path``'s timeline) to silence, with counts."""
    import soundfile as sf

    spans: list[tuple[float, float]] = []
    whole = aligned = 0
    aligner = None
    if use_aligner and available():
        if not _ALIGNER:
            _ALIGNER.append(_Aligner(model_path()))
        aligner = _ALIGNER[0]
    with sf.SoundFile(wav_path) as f:
        for raw, masked in zip(raw_segments, masked_segments):
            m_text = masked.get("text") or ""
            if not any(_TAG.match(w) for w in m_text.split()):
                continue
            start, end = float(raw["start"]), float(raw["end"])
            idx = masked_word_indices(raw.get("text") or "", m_text)
            words = (raw.get("text") or "").split()
            times = None
            if aligner is not None and idx and len(idx) < len(words):
                try:
                    f.seek(int(start * SAMPLE_RATE))
                    audio = f.read(int((end - start) * SAMPLE_RATE), dtype="float32")
                    times = aligner.word_times(audio, words)
                except Exception as exc:  # noqa: BLE001 - fall back to the whole segment
                    log.info("Word alignment failed for one segment (%s); silencing it whole.", exc)
                    times = None
            if times is None or any(times[i] is None for i in idx):
                spans.append((start, end))
                whole += 1
                continue
            for i in idx:
                s, e = times[i]
                spans.append((max(0.0, start + s - MARGIN_S), start + e + MARGIN_S))
            aligned += 1
    return {"spans": _merge(spans), "segments_aligned": aligned, "segments_silenced_whole": whole}


def _merge(spans: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([round(s, 3), round(e, 3)])
    return out


def write(wav_path: str, spans: list[list[float]], out_path: Path) -> Path:
    """Copy ``wav_path`` to ``out_path`` with every span silenced."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    for s, e in spans:
        audio[int(s * sr):int(e * sr)] = 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, subtype="PCM_16")
    out_path.chmod(0o600)
    return out_path
