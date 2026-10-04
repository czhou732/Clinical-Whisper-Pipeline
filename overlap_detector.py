"""Where two people talk at once, from pyannote's segmentation-3.0 model.

MOSS writes one speaker per stretch of speech, so in crosstalk the second
voice is either dropped or folded into the first. Group discussions overlap
far more than one-to-one interviews, and coders need to know where the
transcript is a simplification. This finds those stretches.

The model is pyannote/segmentation-3.0 (MIT licence, Bredin 2023), 1.5 M
parameters: a learned sinc filterbank, four bidirectional LSTM layers and a
7-way "powerset" output for each ~17 ms frame of a 10 s chunk — silence, one
of three local speakers, or one of the three pairs speaking together. The
network is rebuilt here in plain PyTorch, so the app does not need
pyannote.audio; the weights are the published ones, unchanged.
"""

from __future__ import annotations

import logging
import math
import pickle
import types
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("ClinicalWhisper")

MODEL_REPO = "pyannote/segmentation-3.0"
SAMPLE_RATE = 16000
CHUNK_S = 10.0
STEP_S = 2.5
# Powerset classes: 0 silence, 1-3 one speaker, 4-6 two speakers at once.
_OVERLAP_CLASSES = (4, 5, 6)
_SPEECH_CLASSES = (1, 2, 3, 4, 5, 6)
# Chosen on AMI EN2002a-c (F1 0.79); ES2004a-b, untouched: precision 0.81,
# recall 0.59 (evals/reports/group_speakers.md).
THRESHOLD = 0.3


def _build():
    import torch
    from torch import nn
    import torch.nn.functional as F

    class SincFB(nn.Module):
        """asteroid's ParamSincFB (80 filters: 40 cosine + 40 sine, 251 taps)."""

        def __init__(self, n_filters=80, kernel_size=251, stride=10, sr=SAMPLE_RATE,
                     min_low_hz=50.0, min_band_hz=50.0):
            super().__init__()
            self.n_filters, self.kernel_size, self.stride, self.sr = n_filters, kernel_size, stride, sr
            self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
            half = n_filters // 2
            self.low_hz_ = nn.Parameter(torch.zeros(half, 1))
            self.band_hz_ = nn.Parameter(torch.zeros(half, 1))
            self.register_buffer("window_", torch.zeros(kernel_size // 2))
            self.register_buffer("n_", torch.zeros(1, kernel_size // 2))

        def _make(self, low, high, kind):
            band = (high - low)[:, 0]
            ft_low, ft_high = low @ self.n_, high @ self.n_
            if kind == "cos":
                left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2)) * self.window_
                center = 2 * band.view(-1, 1)
                right = torch.flip(left, dims=[1])
            else:
                left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2)) * self.window_
                center = torch.zeros_like(band.view(-1, 1))
                right = -torch.flip(left, dims=[1])
            bp = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
            return bp.view(self.n_filters // 2, 1, self.kernel_size)

        def forward(self, x):
            low = self.min_low_hz + torch.abs(self.low_hz_)
            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sr / 2)
            filters = torch.cat([self._make(low, high, "cos"), self._make(low, high, "sin")], dim=0)
            return F.conv1d(x, filters, stride=self.stride)

    class FilterbankHolder(nn.Module):
        def __init__(self):
            super().__init__()
            self.filterbank = SincFB()

        def forward(self, x):
            return self.filterbank(x)

    class PyanNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.sincnet = nn.Module()
            self.sincnet.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
            self.sincnet.conv1d = nn.ModuleList([FilterbankHolder(), nn.Conv1d(80, 60, 5), nn.Conv1d(60, 60, 5)])
            self.sincnet.norm1d = nn.ModuleList([nn.InstanceNorm1d(80, affine=True),
                                                 nn.InstanceNorm1d(60, affine=True),
                                                 nn.InstanceNorm1d(60, affine=True)])
            self.lstm = nn.LSTM(60, 128, num_layers=4, bidirectional=True, batch_first=True)
            self.linear = nn.ModuleList([nn.Linear(256, 128), nn.Linear(128, 128)])
            self.classifier = nn.Linear(128, 7)

        def forward(self, wav):  # (batch, 1, samples) -> (batch, frames, 7) log-probabilities
            x = self.sincnet.wav_norm1d(wav)
            for c, (conv, norm) in enumerate(zip(self.sincnet.conv1d, self.sincnet.norm1d)):
                x = conv(x)
                if c == 0:
                    x = torch.abs(x)
                x = F.leaky_relu(norm(F.max_pool1d(x, 3, stride=3)))
            x, _ = self.lstm(x.transpose(1, 2))
            for lin in self.linear:
                x = F.leaky_relu(lin(x))
            return F.log_softmax(self.classifier(x), dim=-1)

    return PyanNet()


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    """Reads the checkpoint's tensors; pyannote/lightning metadata objects become stubs."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("pyannote", "pytorch_lightning", "lightning", "asteroid_filterbanks"):
            return type(name, (_Stub,), {})
        if module.split(".")[0] not in ("torch", "collections", "builtins", "numpy", "_codecs"):
            raise pickle.UnpicklingError(f"unexpected class in checkpoint: {module}.{name}")
        return super().find_class(module, name)


def _checkpoint(local_only: bool = True) -> Optional[Path]:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(MODEL_REPO, "pytorch_model.bin", local_files_only=local_only))
    except Exception as exc:  # noqa: BLE001 - absent model: no crosstalk marks, never a failed job
        log.warning("Overlap model unavailable (%s); crosstalk is not marked.", exc)
        return None


class OverlapDetector:
    """Frame-level probability that two people are speaking at once."""

    def __init__(self, model):
        self.model = model
        self.frame_s: Optional[float] = None

    @classmethod
    def load(cls) -> Optional["OverlapDetector"]:
        path = _checkpoint()
        if path is None:
            return None
        import torch

        mod = types.ModuleType("restricted_pickle")
        mod.Unpickler = _Unpickler
        mod.load = lambda f, **kw: _Unpickler(f, **kw).load()
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=mod)
        model = _build()
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        return cls(model)

    def probabilities(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(overlap, speech) probability per frame for 16 kHz mono audio, and the frame length.

        10 s chunks every 2.5 s; each frame's probabilities are averaged over
        the chunks that contain it.
        """
        import torch

        chunk, step = int(CHUNK_S * SAMPLE_RATE), int(STEP_S * SAMPLE_RATE)
        n = len(audio)
        starts = list(range(0, max(n - chunk, 0) + 1, step))
        if not starts or starts[-1] + chunk < n:
            starts.append(max(n - chunk, 0))
        with torch.inference_mode():
            probe = self.model(torch.zeros(1, 1, chunk))
        frames_per_chunk = probe.shape[1]
        frame_s = CHUNK_S / frames_per_chunk
        total = int(math.ceil(n / SAMPLE_RATE / frame_s)) + 1
        ov, sp, cnt = np.zeros(total), np.zeros(total), np.zeros(total)
        batch = 16
        for b in range(0, len(starts), batch):
            group = starts[b:b + batch]
            x = np.stack([np.pad(audio[s:s + chunk], (0, max(0, chunk - len(audio[s:s + chunk]))))
                          for s in group]).astype(np.float32)
            with torch.inference_mode():
                p = self.model(torch.from_numpy(x)[:, None, :]).exp().numpy()
            for s, probs in zip(group, p):
                i0 = int(round(s / SAMPLE_RATE / frame_s))
                m = min(frames_per_chunk, total - i0)
                ov[i0:i0 + m] += probs[:m, list(_OVERLAP_CLASSES)].sum(axis=1)
                sp[i0:i0 + m] += probs[:m, list(_SPEECH_CLASSES)].sum(axis=1)
                cnt[i0:i0 + m] += 1
        cnt[cnt == 0] = 1
        return ov / cnt, sp / cnt, frame_s

    def regions(self, audio: np.ndarray, threshold: float = THRESHOLD, min_s: float = 0.3,
                merge_gap_s: float = 0.2) -> list[tuple[float, float]]:
        """Stretches (start, end) in seconds where overlap probability passes ``threshold``."""
        ov, _, frame_s = self.probabilities(audio)
        on = ov >= threshold
        out: list[list[float]] = []
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                a, b = i * frame_s, j * frame_s
                if out and a - out[-1][1] <= merge_gap_s:
                    out[-1][1] = b
                else:
                    out.append([a, b])
                i = j
            else:
                i += 1
        return [(round(a, 2), round(b, 2)) for a, b in out if b - a >= min_s]


def mark_segments(segments: list[dict], regions: list[tuple[float, float]], min_s: float = 0.5) -> int:
    """Set ``crosstalk`` on segments with at least ``min_s`` (or half) of detected overlap."""
    marked = 0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        shared = sum(max(0.0, min(seg["end"], b) - max(seg["start"], a)) for a, b in regions)
        if dur > 0 and shared >= min(min_s, 0.5 * dur):
            seg["crosstalk"] = True
            marked += 1
    return marked
