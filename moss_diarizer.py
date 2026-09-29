#!/usr/bin/env python3
"""
MOSS Diarizer Module for ClinicalWhisper.

Wraps OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B) for joint transcription and
diarization in a single pass — no separate ASR + Pyannote alignment step.

The load/inference path follows the model card's documented usage:
``AutoModelForCausalLM`` + ``AutoProcessor`` with ``trust_remote_code=True``,
driven through the ``moss_transcribe_diarize`` helper utilities.

Runs on Apple Silicon (MPS), CUDA, or CPU. On Apple Silicon the text decoder
runs on MLX (see ``moss_mlx``), which is several times faster than PyTorch on
MPS and — unlike it — can attend over an hour-long prompt without exhausting
memory. The model is public and ungated —
no Hugging Face token is required — but the ~1.7 GB weights are downloaded to
``~/.cache/huggingface`` on first use.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch

import moss_windowed

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
except ImportError:  # pragma: no cover - exercised only when deps are missing
    AutoModelForCausalLM = None
    AutoProcessor = None
    DEFAULT_PROMPT = ""

log = logging.getLogger("ClinicalWhisper")

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# MOSS encodes audio at 12.5 prompt tokens per second, and speech-dense audio
# transcribes to ~4-5 output tokens per second. The old fixed cap of 8192 new
# tokens silently truncated anything longer than roughly half an hour.
_OUTPUT_TOKENS_PER_SECOND = 10
_MIN_NEW_TOKENS = 4096
_CONTEXT_LIMIT = 131072

class _Cancelled(Exception):
    """Raised out of the token callback to stop generation early."""


def build_prompt(base_prompt: str, hotwords: Optional[list]) -> str:
    """Append domain hotwords in the form the model card documents.

    Clinical audio is full of drug names and instrument names that a general
    transcriber mangles; MOSS accepts a hotword hint appended to the prompt.
    """
    terms = [str(w).strip() for w in (hotwords or []) if str(w).strip()]
    if not terms:
        return base_prompt
    return f"{base_prompt}热词提示：{', '.join(terms)}"


def _audio_duration(path: str) -> Optional[float]:
    """Duration in seconds, for turning a token count into a percentage."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None

# Cache: the 0.9B weights take ~10 s to load, so keep them resident across the
# files of one batch rather than reloading per file. It must be evictable — held
# unconditionally, MOSS (~1.8 GB at float16) sat in memory for the life of the
# app and competed with the scoring model for unified memory.
_MODEL_CACHE: dict[tuple[str, str, str], tuple] = {}


def unload_models() -> int:
    """Drop cached MOSS weights. Returns how many entries were freed."""
    import gc

    freed = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    try:
        import moss_mlx

        freed += moss_mlx.unload()
    except ImportError:  # pragma: no cover - mlx absent
        pass
    gc.collect()

    # Return the MPS allocator's cached blocks to the OS, otherwise the freed
    # weights stay counted against the process on Apple Silicon.
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover - best effort
        pass

    if freed:
        log.info("Released MOSS model from memory.")
    return freed


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
        max_new_tokens: Optional[int] = None,
        prompt: Optional[str] = None,
        dtype: Optional[str] = None,
        hotwords: Optional[list] = None,
        backend: Optional[str] = None,
        mlx_dtype: str = "float16",
        window_seconds: float = 300.0,
        window_overlap_seconds: float = 30.0,
        batch_size: int = 32,
        speaker_similarity: Optional[float] = None,
        speaker_merge_similarity: Optional[float] = None,
        speaker_split_similarity: Optional[float] = None,
        num_speakers: Optional[int] = None,
    ):
        self.model = None
        self.processor = None
        self.is_available = False
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.window_seconds = window_seconds
        self.window_overlap_seconds = window_overlap_seconds
        self.batch_size = max(1, int(batch_size))
        self.speaker_similarity = speaker_similarity
        self.speaker_merge_similarity = speaker_merge_similarity
        self.speaker_split_similarity = speaker_split_similarity
        # Known speaker count (e.g. 2 for a clinical interview); extra labels
        # are folded into this many. None leaves the count to the model.
        self.num_speakers = int(num_speakers) if num_speakers else None
        self._window_template: Optional[str] = None
        self.prompt = build_prompt(prompt or DEFAULT_PROMPT, hotwords)

        if AutoModelForCausalLM is None or AutoProcessor is None:
            log.warning(
                "moss-transcribe-diarize / transformers not installed. MOSS is disabled."
            )
            return

        self.device = select_device(device)
        self.dtype = select_dtype(self.device, dtype)
        self.backend = self._select_backend(backend)

        if self.backend == "mlx":
            try:
                import moss_mlx

                self.processor = self._processor(model_name)
                self._mlx = moss_mlx.MOSSOnMLX.get(model_name, mlx_dtype)
                self.is_available = True
                log.info("MOSS model ready on MLX (%s).", mlx_dtype)
                return
            except Exception as e:
                log.warning("MLX unavailable (%s); falling back to PyTorch.", e)
                self.backend = "torch"

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

    @staticmethod
    def _processor(model_name: str):
        key = ("processor", model_name, "")
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return _MODEL_CACHE[key]

    def _select_backend(self, requested: Optional[str]) -> str:
        """``mlx`` on Apple Silicon when available, otherwise ``torch``."""
        requested = (requested or "auto").lower()
        if requested == "torch":
            return "torch"
        try:
            import moss_mlx

            available = moss_mlx.mlx_available()
        except ImportError:
            available = False
        if requested == "mlx" and not available:
            raise RuntimeError("MLX backend requested but MLX is not available.")
        return "mlx" if available and self.device.type != "cuda" else "torch"

    def _token_budget(self, audio_seconds: Optional[float], prompt_len: int) -> int:
        if self.max_new_tokens is not None:
            budget = self.max_new_tokens
        else:
            budget = max(_MIN_NEW_TOKENS, int((audio_seconds or 0) * _OUTPUT_TOKENS_PER_SECOND))
        return max(1, min(budget, _CONTEXT_LIMIT - prompt_len))

    def _prepare_window(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        return moss_windowed.prepare_window(self, audio)

    def _batch_limit(self, prompt_len: int, expected_new: int) -> int:
        return moss_windowed.batch_limit(self, prompt_len, expected_new)

    def _transcribe_windowed(self, audio_path: str, token_callback) -> list[dict]:
        return moss_windowed.transcribe_windowed(self, audio_path, token_callback)

    def process_files(
        self,
        audio_paths: list[str],
        progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[list[dict]]:
        """Transcribe several files together; windows share batches across files.

        Returns one segment list per file (possibly empty for silent audio).
        """
        if not self.is_available:
            raise RuntimeError("MOSS model is not available.")

        def _on_token(count: int) -> None:
            if should_cancel is not None and should_cancel():
                raise _Cancelled()
            if progress_cb is not None:
                progress_cb(count, None)

        callback = _on_token if (progress_cb or should_cancel) else None
        try:
            return moss_windowed.transcribe_many(self, list(audio_paths), callback)
        except _Cancelled:
            log.info("Transcription cancelled.")
            raise
        except Exception as e:
            log.error("Error during MOSS inference: %s", e)
            raise RuntimeError(f"MOSS inference failed: {e}") from e

    def process_file(
        self,
        audio_path: str,
        progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[dict]:
        """Transcribe and diarize one audio file.

        ``progress_cb(tokens_generated, audio_seconds)`` is called as the model
        emits tokens. This is by far the longest stage — on a one-hour interview
        it runs for minutes — and without it the UI shows a single frozen line
        the whole time.

        ``should_cancel()`` is polled during generation; returning True aborts.

        Returns:
            Segments as ``[{"start": float, "end": float, "speaker": str,
            "text": str}, ...]``. Speaker labels are MOSS's anonymous ``S01``,
            ``S02``, ... tags.

        Raises:
            RuntimeError: if the model is unavailable or inference fails.
        """
        if not self.is_available:
            raise RuntimeError("MOSS model is not available.")

        log.info("Running MOSS transcribe + diarize on %s...", audio_path)

        audio_seconds = _audio_duration(audio_path)

        def _on_token(count: int) -> None:
            if should_cancel is not None and should_cancel():
                raise _Cancelled()
            if progress_cb is not None:
                progress_cb(count, audio_seconds)

        callback = _on_token if (progress_cb or should_cancel) else None
        # Windowed on every backend: one pass over a long file slows as context
        # grows and, on a 29-min meeting, silently dropped the last 3 minutes.
        try:
            segments = self._transcribe_windowed(audio_path, callback)
        except _Cancelled:
            log.info("Transcription cancelled.")
            raise
        except Exception as e:
            log.error("Error during MOSS inference: %s", e)
            raise RuntimeError(f"MOSS inference failed: {e}") from e

        if not segments:
            # The model emits EOS immediately on silent or near-silent audio.
            raise RuntimeError(
                "MOSS returned an empty transcript — the audio may be silent, "
                "too quiet, or not speech."
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
