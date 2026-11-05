"""MOSS's audio side on MLX: Whisper-medium encoder, 4x time merge, adaptor.

With this and :mod:`moss_mlx` the whole model runs on MLX, so the PyTorch copy
of MOSS (~1.8 GB) never has to load on Apple Silicon. On MPS the encoder cost
~7 s per 5 minutes of audio — about 85 s for an hour, over half of a
two-minute budget on its own.

Weights come straight from MOSS's safetensors file. The layer names mirror
Hugging Face's ``WhisperEncoder`` and MOSS's ``VQAdaptor`` so the mapping is
mechanical: only the Conv1d kernels need transposing.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_ENCODER_PREFIX = "model.whisper_encoder."
_ADAPTOR_PREFIX = "model.vq_adaptor."

# Chunks of 30 s encoded per call. Attention over 1500 frames x 16 heads is
# ~72 MB per chunk per layer at float16; 16 keeps the peak modest.
_ENCODE_BATCH = 16


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, d = x.shape

        def split(t):
            return t.reshape(b, n, self.heads, -1).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)),
            scale=self.scale,
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, n, d))


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.self_attn = _SelfAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class _WhisperEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["d_model"]
        self.conv1 = nn.Conv1d(cfg["num_mel_bins"], dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg["max_source_positions"], dim)
        self.layers = [
            _EncoderLayer(dim, cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"])
            for _ in range(cfg["encoder_layers"])
        ]
        self.layer_norm = nn.LayerNorm(dim)

    def __call__(self, mel: mx.array) -> mx.array:
        """``(B, frames, mels)`` log-mel -> ``(B, frames // 2, dim)``."""
        x = nn.gelu(self.conv1(mel))
        x = nn.gelu(self.conv2(x))
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class _Adaptor(nn.Module):
    """``Linear -> SiLU -> Linear -> LayerNorm``; indices match MOSS's Sequential."""

    def __init__(self, in_dim: int, dim: int, eps: float):
        super().__init__()
        self.layers = [nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim, eps=eps)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MLXAudioEncoder:
    """Turns MOSS processor output into the audio embeddings the decoder reads."""

    def __init__(self, model_dir: Path, dtype: str = "float16"):
        config = json.loads((model_dir / "config.json").read_text())
        self.merge = int(config["audio_merge_size"])
        self.encoder = _WhisperEncoder(config["audio_config"])
        self.adaptor = _Adaptor(
            config["adaptor_input_dim"],
            config["text_config"]["hidden_size"],
            config["text_config"]["rms_norm_eps"],
        )

        target = getattr(mx, dtype)
        enc, ada = [], []
        for shard in sorted(model_dir.glob("*.safetensors")):
            for name, t in mx.load(str(shard)).items():
                if name.startswith(_ENCODER_PREFIX):
                    key = name[len(_ENCODER_PREFIX):]
                    if key.startswith("conv") and key.endswith("weight"):
                        t = t.transpose(0, 2, 1)  # (out, in, k) -> (out, k, in)
                    enc.append((key, t.astype(target)))
                elif name.startswith(_ADAPTOR_PREFIX):
                    ada.append((name[len(_ADAPTOR_PREFIX):], t.astype(target)))
        self.encoder.load_weights(enc, strict=True)
        self.adaptor.load_weights(ada, strict=True)
        self.encoder.eval()
        self.adaptor.eval()
        mx.eval(self.encoder.parameters(), self.adaptor.parameters())
        self.dtype = target

    def encode(
        self,
        input_features: np.ndarray,
        token_lengths: np.ndarray,
        chunk_owner: np.ndarray,
        n_items: int,
    ) -> list[mx.array]:
        """Encode every 30 s chunk and regroup them per audio item.

        Args:
            input_features: ``(chunks, mels, frames)`` from the MOSS processor.
            token_lengths: audio tokens each chunk contributes (after merging).
            chunk_owner: which audio item each chunk belongs to.
            n_items: number of audio items.

        Returns:
            One ``(tokens, hidden)`` array per item.
        """
        mel = mx.array(input_features.transpose(0, 2, 1)).astype(self.dtype)
        hidden = []
        for s in range(0, mel.shape[0], _ENCODE_BATCH):
            h = self.encoder(mel[s:s + _ENCODE_BATCH])
            mx.eval(h)
            hidden.append(h)
        hidden = mx.concatenate(hidden, axis=0)

        out = []
        for item in range(n_items):
            parts = [
                hidden[c, : int(token_lengths[c]) * self.merge]
                for c in np.nonzero(chunk_owner == item)[0]
            ]
            feats = mx.concatenate(parts, axis=0)
            usable = (feats.shape[0] // self.merge) * self.merge
            merged = feats[:usable].reshape(usable // self.merge, -1)
            out.append(self.adaptor(merged))
        return out
