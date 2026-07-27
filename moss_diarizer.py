#!/usr/bin/env python3
"""
MOSS Diarizer Module for ClinicalWhisper.

Wraps OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B) for joint transcription and
diarization in a single pass — no separate ASR + Pyannote alignment step.

The load/inference path follows the model card's documented usage:
``AutoModelForCausalLM`` + ``AutoProcessor`` with ``trust_remote_code=True``,
driven through the ``moss_transcribe_diarize`` helper utilities.

Runs on Apple Silicon (MPS), CUDA, or CPU. The model is public and ungated —
no Hugging Face token is required — but the ~1.7 GB weights are downloaded to
``~/.cache/huggingface`` on first use.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        DEFAULT_PROMPT,
        build_transcription_messages,
        generate_transcription,
    )
except ImportError:  # pragma: no cover - exercised only when deps are missing
    AutoModelForCausalLM = None
    AutoProcessor = None
    parse_transcript = None
    build_transcription_messages = None
    generate_transcription = None
    DEFAULT_PROMPT = ""

log = logging.getLogger("ClinicalWhisper")

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# Module-level cache: the 0.9B weights take ~10 s to load, so keep them resident
# across jobs within a single process rather than reloading per file.
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the best available accelerator.

    ``moss_transcribe_diarize.inference_utils.resolve_device('auto')`` only
    considers CUDA, so it returns CPU on Apple Silicon. This picks MPS there,
    which measured ~2x faster than CPU on an M-series machine.
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def select_dtype(device: torch.device, preferred: Optional[str] = None) -> torch.dtype:
    """Pick the compute dtype for MOSS.

    On MPS, float16 measured ~2.4x faster than float32 (15.2s -> 6.2s on a 30s
    clip) and produced a byte-identical transcript, while halving resident
    memory so the LLM stage is not competing for unified memory. Set
    ``moss.dtype: float32`` in config.yaml to fall back if a recording ever
    produces degraded output.
    """
    if preferred and preferred != "auto":
        key = preferred.lower()
        if key in _DTYPES:
            return _DTYPES[key]
        log.warning("Unknown moss.dtype %r — falling back to auto.", preferred)

    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


class MOSSDiarizer:
    """Joint transcription + diarization via MOSS-Transcribe-Diarize."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        max_new_tokens: int = 8192,
        prompt: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.model = None
        self.processor = None
        self.is_available = False
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt or DEFAULT_PROMPT

        if AutoModelForCausalLM is None or generate_transcription is None:
            log.warning(
                "moss-transcribe-diarize / transformers not installed. MOSS is disabled."
            )
            return

        self.device = select_device(device)
        self.dtype = select_dtype(self.device, dtype)

        cache_key = (model_name, str(self.device), str(self.dtype))
        try:
            if cache_key in _MODEL_CACHE:
                self.model, self.processor = _MODEL_CACHE[cache_key]
            else:
                log.info("Loading MOSS model %s on %s...", model_name, self.device)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, dtype="auto"
                )
                model = model.to(dtype=self.dtype).to(self.device).eval()
                processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                _MODEL_CACHE[cache_key] = (model, processor)
                self.model, self.processor = model, processor
            self.is_available = True
            log.info("MOSS model ready on %s.", self.device)
        except Exception as e:
            log.error("Failed to load MOSS model: %s", e)

    def process_file(self, audio_path: str) -> list[dict]:
        """Transcribe and diarize one audio file.

        Returns:
            Segments as ``[{"start": float, "end": float, "speaker": str,
            "text": str}, ...]``. Speaker labels are MOSS's anonymous ``S01``,
            ``S02``, ... tags.

        Raises:
            RuntimeError: if the model is unavailable or inference fails.
        """
        if not self.is_available or self.model is None:
            raise RuntimeError("MOSS model is not available.")

        log.info("Running MOSS transcribe + diarize on %s...", audio_path)
        try:
            messages = build_transcription_messages(audio_path, self.prompt)
            result = generate_transcription(
                self.model,
                self.processor,
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                device=self.device,
                dtype=self.dtype,
            )
        except Exception as e:
            log.error("Error during MOSS inference: %s", e)
            raise RuntimeError(f"MOSS inference failed: {e}") from e

        raw_text = (result.get("text") or "").strip()
        if not raw_text:
            # The model emits EOS immediately on silent or near-silent audio.
            raise RuntimeError(
                "MOSS returned an empty transcript — the audio may be silent, "
                "too quiet, or not speech."
            )

        segments = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": seg.speaker or "UNKNOWN",
                "text": seg.text or "",
            }
            for seg in parse_transcript(raw_text)
        ]

        if not segments:
            raise RuntimeError(
                "MOSS produced output that could not be parsed into segments."
            )

        log.info("MOSS produced %d segments.", len(segments))
        return segments


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) > 1:
        diarizer = MOSSDiarizer()
        for r in diarizer.process_file(sys.argv[1]):
            print(f"[{r['start']:.2f}s -> {r['end']:.2f}s] {r['speaker']}: {r['text']}")
    else:
        print("Usage: python moss_diarizer.py <audio_file>")
