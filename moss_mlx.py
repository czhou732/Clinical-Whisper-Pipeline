"""Run MOSS on MLX: the Qwen3 text decoder here, the audio side in moss_mlx_audio.

MOSS-Transcribe-Diarize is a Whisper-medium audio encoder feeding a Qwen3-0.6B
text decoder. The encoder runs once over the whole file and parallelises well
on MPS. The decoder runs once per output token — roughly 20k times for an hour
of audio — and that loop is where nearly all of the transcription time goes.
PyTorch on MPS pays a large fixed cost per decode step; MLX does not.

Both halves load directly from MOSS's safetensors file, so the PyTorch copy
of the model never loads on Apple Silicon. Decoding is greedy, matching the
PyTorch path, and float16 — bfloat16 measurably changed transcripts, dropping
filler words the hesitancy measures depend on.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import mlx.core as mx
    from mlx_lm.models import qwen3
except ImportError:  # non-Apple platforms, or mlx-lm not installed
    mx = None

_LM_PREFIX = "model.language_model."

# One loaded model per (weights dir, dtype), shared across files in a batch.
_CACHE: dict[tuple[str, str], "MOSSOnMLX"] = {}


def mlx_available() -> bool:
    """True on Apple Silicon with mlx-lm importable."""
    return (
        mx is not None
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
    )


def resolve_model_dir(model_name: str) -> Path:
    """Return the local directory holding ``config.json`` and the weights."""
    local = Path(model_name).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name, local_files_only=True))


class MLXDecoder:
    """The Qwen3 decoder half of MOSS, running on MLX."""

    def __init__(self, model_dir: Path, dtype: str = "float16"):
        if not mlx_available():
            raise RuntimeError("MLX is not available on this machine.")

        config = json.loads((model_dir / "config.json").read_text())
        self.model = qwen3.Model(qwen3.ModelArgs.from_dict(config["text_config"]))

        weights = {}
        for shard in sorted(model_dir.glob("*.safetensors")):
            for name, tensor in mx.load(str(shard)).items():
                if name.startswith(_LM_PREFIX):
                    weights["model." + name[len(_LM_PREFIX):]] = tensor
        if not weights:
            raise RuntimeError(f"No decoder weights found under {model_dir}.")

        target = getattr(mx, dtype)
        self.model.load_weights(
            [(k, v.astype(target)) for k, v in weights.items()], strict=True
        )
        self.model.eval()
        mx.eval(self.model.parameters())
        self.dtype = dtype

    def generate_batch(
        self,
        input_ids: np.ndarray,
        inputs_embeds: "mx.array",
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        expected_new_tokens: Optional[int] = None,
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> list[list[int]]:
        """Greedy-decode ``B`` equal-length prompts in lockstep.

        Batching reads the decoder weights once per step for every sequence
        instead of once per sequence, which is where the throughput comes
        from. Prompts must share one length (see ``moss_chunking``), so no
        padding or per-row masking is needed.

        Memory is the constraint that decides speed here: a batch that spills
        into swap runs ~15x slower (measured 0.39 s/step vs ~25 ms). So:

        * each layer's KV cache is sized once for prompt + ``expected_new_tokens``
          instead of regrowing by concatenation every 256 tokens;
        * rows that finish are dropped from the batch, so late steps stop
          paying for windows that are already done.

        Args:
            input_ids: ``(B, L)`` prompt ids.
            inputs_embeds: ``(B, L, hidden)`` MLX prompt embeddings, audio injected.
            token_callback: called with the running total of generated tokens;
                may raise to abort.

        Returns:
            One list of generated ids per row, stop token excluded.
        """
        from mlx_lm.models.cache import KVCache

        batch, length = input_ids.shape
        ids = mx.array(input_ids.astype(np.int32))
        embeds = inputs_embeds.astype(getattr(mx, self.dtype))

        expected = expected_new_tokens or max_new_tokens
        capacity = -(-(length + expected) // 256) * 256
        cache = [KVCache() for _ in self.model.layers]
        for c in cache:
            c.step = capacity  # one allocation; growth past it falls back to 256-token steps

        prefill_step = 2048
        for s in range(0, length - 1, prefill_step):
            e = min(s + prefill_step, length - 1)
            self.model(ids[:, s:e], cache=cache, input_embeddings=embeds[:, s:e])
            mx.eval([c.state for c in cache])
        logits = self.model(ids[:, -1:], cache=cache, input_embeddings=embeds[:, -1:])
        y = mx.argmax(logits[:, -1, :], axis=-1)
        del ids, embeds, logits
        for c in cache:
            c.step = 256
        mx.clear_cache()

        out: list[list[int]] = [[] for _ in range(batch)]
        rows = list(range(batch))  # original index of each row still decoding
        pending: list[mx.array] = []
        produced = 0
        for step in range(max_new_tokens):
            pending.append(y)
            nxt = mx.argmax(self.model(y[:, None], cache=cache)[:, -1, :], axis=-1)
            mx.async_eval(nxt)
            if len(pending) < 32 and step < max_new_tokens - 1:
                y = nxt
                continue

            # Every 32 steps: collect tokens and retire finished rows. Syncing
            # every step would stall the GPU.
            block = np.array(mx.stack(pending, axis=1))
            pending = []
            keep = []
            for i, orig in enumerate(rows):
                for t in block[i].tolist():
                    if t in eos_token_ids:
                        break
                    out[orig].append(t)
                    produced += 1
                else:
                    keep.append(i)
            if token_callback is not None:
                token_callback(produced)
            if not keep:
                break
            if len(keep) < len(rows):
                idx = mx.array(keep)
                for c in cache:
                    c.keys, c.values = c.keys[idx], c.values[idx]
                nxt = nxt[idx]
                rows = [rows[i] for i in keep]
                mx.eval([c.state for c in cache])
                mx.clear_cache()
            elif step % 256 < 32:
                mx.clear_cache()
            y = nxt

        del cache
        mx.clear_cache()
        return out


class MOSSOnMLX:
    """Everything MOSS needs at inference time, on MLX."""

    def __init__(self, model_dir: Path, dtype: str = "float16"):
        from moss_mlx_audio import MLXAudioEncoder

        config = json.loads((model_dir / "config.json").read_text())
        generation = json.loads((model_dir / "generation_config.json").read_text())
        eos = generation["eos_token_id"]
        self.eos_token_ids = set(eos) if isinstance(eos, list) else {eos}
        self.audio_token_id = int(config["audio_token_id"])
        self.text_config = config["text_config"]
        self.decoder = MLXDecoder(model_dir, dtype)
        self.audio = MLXAudioEncoder(model_dir, dtype)

    @classmethod
    def get(cls, model_name: str, dtype: str = "float16") -> "MOSSOnMLX":
        model_dir = resolve_model_dir(model_name)
        key = (str(model_dir), dtype)
        if key not in _CACHE:
            log.info("Loading MOSS on MLX (%s) from %s...", dtype, model_dir)
            _CACHE[key] = cls(model_dir, dtype)
        return _CACHE[key]

    def prompt_embeddings(self, input_ids: np.ndarray, audio: list["mx.array"]) -> "mx.array":
        """Token embeddings with each row's audio written over its placeholders.

        Rows come from equal-length windows, so the placeholders sit at the
        same positions in every row.
        """
        slots = input_ids == self.audio_token_id
        if not (slots == slots[0]).all():
            raise ValueError("windows must share one prompt layout")
        positions = mx.array(np.nonzero(slots[0])[0])
        embeds = self.decoder.model.model.embed_tokens(mx.array(input_ids.astype(np.int32)))
        embeds[:, positions] = mx.stack(audio).astype(embeds.dtype)
        return embeds


def unload() -> int:
    """Drop cached MLX models and return MLX's buffer cache to the OS."""
    n = len(_CACHE)
    _CACHE.clear()
    if mx is not None:
        mx.clear_cache()
    return n
