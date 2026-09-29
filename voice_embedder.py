"""Speaker voice embeddings for linking speakers across transcription windows.

MOSS labels speakers per window, so the windows must be stitched together by
voice. Averaged MOSS encoder features did that poorly: on a 76-minute journal
club, two *different* people reached cosine similarity 0.68 while one person
matched themselves at 0.84 — too little margin to set a safe threshold. This
uses WeSpeaker's ResNet34 (VoxCeleb, CC BY 4.0), a model trained specifically
to tell voices apart, run through ONNX Runtime (CoreML where available).

Preprocessing follows WeSpeaker's own recipe: waveform scaled to 16-bit range,
80-bin Kaldi filterbank (25 ms / 10 ms), no dither at inference, per-utterance
mean normalisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("ClinicalWhisper")

MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
SAMPLE_RATE = 16000

# Segments shorter than this carry too little voice to embed reliably. Capping
# each speaker at 20 s instead of 60 s cost separation on AMI (same-speaker
# minimum fell from 0.75 to 0.39, below the different-speaker maximum), so 60.
_MIN_SEGMENT_S = 0.5
_MAX_SPEECH_S = 60.0


class VoiceEmbedder:
    """256-dimensional voice embeddings from 16 kHz mono audio."""

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        # CoreML ran 14x faster than the CPU with identical embeddings
        # (0.5 s vs 7.2 s for 8 one-minute clips); CPU remains the fallback.
        providers = [
            p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def load(cls) -> Optional["VoiceEmbedder"]:
        """The embedder, or None if the model is not available offline."""
        try:
            from huggingface_hub import hf_hub_download

            return cls(Path(hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=True)))
        except Exception as e:  # missing model or runtime: fall back, never fail the job
            log.warning("Voice embedding model unavailable (%s); linking by encoder features.", e)
            return None

    @staticmethod
    def _fbank(audio: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        wave = torch.from_numpy(audio.astype(np.float32) * (1 << 15)).unsqueeze(0)
        feats = kaldi.fbank(
            wave,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        return (feats - feats.mean(dim=0)).numpy()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        feats = self._fbank(audio)[None].astype(np.float32)
        return self.session.run(None, {self.input_name: feats})[0][0]

    def speakers(
        self, segments: list[dict], audio: np.ndarray, offset_s: float = 0.0
    ) -> dict[str, tuple[np.ndarray, float]]:
        """One embedding and the talk time per speaker in ``segments``.

        ``segments`` carry times relative to ``audio``'s start plus
        ``offset_s``. Each speaker's longest segments are concatenated, up to
        :data:`_MAX_SPEECH_S`, and embedded together.
        """
        by_speaker: dict[str, list[dict]] = {}
        talk: dict[str, float] = {}
        for seg in segments:
            dur = seg["end"] - seg["start"]
            talk[seg["speaker"]] = talk.get(seg["speaker"], 0.0) + max(dur, 0.0)
            if dur >= _MIN_SEGMENT_S:
                by_speaker.setdefault(seg["speaker"], []).append(seg)

        out: dict[str, tuple[np.ndarray, float]] = {}
        for spk, segs in by_speaker.items():
            pieces, total = [], 0.0
            for seg in sorted(segs, key=lambda s: s["start"] - s["end"]):  # longest first
                a = int((seg["start"] - offset_s) * SAMPLE_RATE)
                b = int((seg["end"] - offset_s) * SAMPLE_RATE)
                clip = audio[max(a, 0):max(b, 0)]
                if len(clip) < _MIN_SEGMENT_S * SAMPLE_RATE:
                    continue
                pieces.append(clip)
                total += len(clip) / SAMPLE_RATE
                if total >= _MAX_SPEECH_S:
                    break
            if pieces:
                out[spk] = (self.embed(np.concatenate(pieces)), talk[spk])
        return out
