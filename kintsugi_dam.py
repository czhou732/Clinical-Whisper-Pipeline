"""Kintsugi's Depression-Anxiety Model (DAM 3.1) as an optional add-on.

Kintsugi Health trained this on speech from about 35,000 people (863 hours)
with clinician-administered and self-reported PHQ-9 and GAD-7, then released
it under Apache-2.0 when the company closed (KintsugiHealth/dam on Hugging
Face). It reads only the sound of the voice, not the words, and returns
severity levels in PHQ-9 and GAD-7 bands. Their stated envelope: one voice,
English, at least 30 s of speech, quiet recording.

What this port changes, and why:

* The model is rebuilt here from their published architecture
  (model.py/config.py/featex.py, Apache-2.0, Kintsugi Health), with their
  low-rank fine-tuning folded into the weights. That removes the PEFT
  dependency and any download of the base Whisper model: the checkpoint
  already carries every weight, and loading is strict, so a mismatch fails.
* The checkpoint is loaded with ``weights_only=True`` (no code in a pickle
  can run).
* It is fed only the participant's own speech, cut from the diarized
  recording, because the model expects a single voice.

Results are research estimates: the model was not validated on this
population or on clinical interview recordings, and it is not a diagnosis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import bundled_models

log = logging.getLogger("ClinicalWhisper")

ROOT = bundled_models.APP_SUPPORT / "addons" / "kintsugi"
CHECKPOINT = "dam3.1.ckpt"
SAMPLE_RATE = 16000
CHUNK_S = 30
MIN_SPEECH_S = 30.0

# From their config.py: score thresholds chosen on their validation set.
THRESHOLDS = {"depression": [-0.6699, -0.2908], "anxiety": [-0.7939, -0.2173, 0.1521]}
LEVELS = {
    "depression": ["none (PHQ-9 0-9)", "mild to moderate (PHQ-9 10-14)", "severe (PHQ-9 15+)"],
    "anxiety": ["none (GAD-7 0-4)", "mild (GAD-7 5-9)", "moderate (GAD-7 10-14)",
                "severe (GAD-7 15+)"],
}
# Average log-mel energies the model was trained to expect (their config.py).
_LOGMEL_ENERGIES = [
    0.34912264, 0.58558977, 0.7912451, 0.92767584, 0.98273695, 0.98439455, 0.9603633,
    0.93906444, 0.9366281, 0.93200225, 0.916437, 0.8928787, 0.8637211, 0.83265126,
    0.79977655, 0.7778334, 0.7561299, 0.72997606, 0.70391226, 0.6800474, 0.65755,
    0.63536274, 0.61355984, 0.5923383, 0.5720056, 0.55244887, 0.53684795, 0.5221597,
    0.5098636, 0.49923953, 0.48908615, 0.47840047, 0.46758702, 0.47343993, 0.46268672,
    0.4475126, 0.46747103, 0.45131385, 0.4635319, 0.44889897, 0.45491976, 0.4373785,
    0.43154317, 0.42194438, 0.41158468, 0.40096927, 0.3933149, 0.38795966, 0.38441542,
    0.38454026, 0.3815766, 0.3768835, 0.3719921, 0.3654539, 0.35399568, 0.3425986,
    0.32823247, 0.31404305, 0.30564603, 0.29617435, 0.29273877, 0.28560263, 0.27459458,
    0.26876706, 0.25825337, 0.24759005, 0.24090728, 0.2344712, 0.22529823, 0.20880115,
    0.193578, 0.18290243, 0.17621627, 0.17087021, 0.16641389, 0.15932252, 0.14312662,
    0.11790597, 0.08030523, 0.03747071,
]
_LORA_SCALE = 64.0 / 32.0  # lora_alpha / r in their config

# "Can't tell" band for PHQ-9 >= 10, chosen on their released validation scores
# with a 20% budget (evals/kintsugi_analysis.py). On their test set: sensitivity
# 0.71 and specificity 0.74 on the decided cases, 19% undecided; their single
# threshold gave 0.58 / 0.78.
SCREEN_BAND = (-1.033, -0.654)
PERFORMANCE_NOTE = (
    "On Kintsugi's own test data (n = 7,034, PHQ-9 >= 10): AUC 0.76. It misses more "
    "depression in some groups: sensitivity 0.16 in adults over 60, 0.45 in men, 0.41 in one "
    "Black respondent group, against 0.58 overall; and is less accurate on noisy recordings. "
    "It follows low mood more than loss of interest. See evals/reports/kintsugi_validation.md.")


def checkpoint_path(root: Optional[Path] = None) -> Optional[Path]:
    """The installed checkpoint (add-on), else a Hugging Face cache copy."""
    root = root or ROOT
    if (root / CHECKPOINT).is_file():
        return root / CHECKPOINT
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache("KintsugiHealth/dam", CHECKPOINT)
        return Path(hit) if isinstance(hit, str) else None
    except ImportError:  # pragma: no cover
        return None


def available() -> bool:
    return checkpoint_path() is not None


def _merge(state: dict, prefix: str) -> dict:
    """One plain Whisper-encoder state dict from a backbone in the checkpoint.

    Low-rank pairs (lora_A, lora_B) are folded into their base weight;
    PEFT's saved copies of fully-trained modules replace the originals.
    """
    import torch

    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    sub = {k[len("base_model.model."):] if k.startswith("base_model.model.") else k: v
           for k, v in sub.items()}
    out: dict = {}
    for key, value in sub.items():
        if ".lora_" in key or ".original_module." in key:
            continue
        name = key.replace(".base_layer.", ".").replace(".modules_to_save.default.", ".")
        out[name] = value
    for key in sub:
        if ".lora_A.default.weight" not in key:
            continue
        base = key.replace(".lora_A.default.weight", "")
        a, b = sub[key], sub[base + ".lora_B.default.weight"]
        out[base + ".weight"] = out[base + ".weight"] + _LORA_SCALE * (b @ a)
    return {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in out.items()}


class _Model:
    """Two Whisper-small encoders -> mean pool -> shared layers -> two heads."""

    def __init__(self, ckpt: Path):
        import torch
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        # whisper-small.en encoder shape; strict loading below checks every tensor.
        cfg = WhisperConfig(d_model=768, encoder_layers=12, encoder_attention_heads=12,
                            encoder_ffn_dim=3072, num_mel_bins=80, max_source_positions=1500,
                            dropout=0.0, activation_dropout=0.0, encoder_layerdrop=0.0)
        self.encoders = []
        for name in ("audio", "llma"):  # their ModuleDict order (sorted keys)
            enc = WhisperEncoder(cfg)
            enc.load_state_dict(_merge(state, f"backbone.{name}.backbone."), strict=True)
            enc.eval()
            self.encoders.append(enc)
        h = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        self.shared = [(h["shared_layers.shared_layers.0.weight"], h["shared_layers.shared_layers.0.bias"]),
                       (h["shared_layers.shared_layers.2.weight"], h["shared_layers.shared_layers.2.bias"])]
        self.heads = {t: (h[f"classifier_head.{t}.linear.weight"], h[f"classifier_head.{t}.linear.bias"],
                          h[f"classifier_head.{t}.final_layer.weight"]) for t in ("depression", "anxiety")}
        self.torch = torch

    def __call__(self, features):
        torch = self.torch
        F = torch.nn.functional
        with torch.no_grad():
            pooled = torch.cat([enc(features).last_hidden_state.mean(dim=1).mean(dim=0, keepdim=True)
                                for enc in self.encoders], dim=1)
            x = F.mish(F.linear(pooled, *self.shared[0]))
            x = F.linear(x, *self.shared[1])
            out = {}
            for task, (w, b, w2) in self.heads.items():
                out[task] = float(F.linear(F.mish(F.linear(x, w, b)), w2)[0, 0])
        return out


_MODEL: list = []


def _features(audio: np.ndarray):
    """Their preprocessing: DC removal, peak normalisation, 30 s chunks, log-mel rescaling."""
    import torch
    from transformers import WhisperFeatureExtractor

    audio = audio.astype(np.float32)
    audio = audio - audio.mean()
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak
    chunk = SAMPLE_RATE * CHUNK_S
    n = int(np.ceil(len(audio) / chunk))
    audio = np.pad(audio, (0, n * chunk - len(audio)))
    extractor = WhisperFeatureExtractor()  # whisper-small.en: 80 mels, 30 s
    feats = extractor([audio[i * chunk:(i + 1) * chunk] for i in range(n)], return_tensors="pt",
                      sampling_rate=SAMPLE_RATE, do_normalize=True).input_features
    energies = torch.tensor(_LOGMEL_ENERGIES)
    feats = feats + (energies.unsqueeze(0) - feats.mean(dim=-1)).unsqueeze(2)
    return feats


def _level(task: str, score: float) -> int:
    return int(np.searchsorted(THRESHOLDS[task], score, side="left"))


def screen(depression_score: float) -> str:
    """"likely PHQ-9 >= 10", "likely below 10" or "can't tell" (the abstain band)."""
    low, high = SCREEN_BAND
    if depression_score > high:
        return "likely PHQ-9 10 or more"
    if depression_score <= low:
        return "likely PHQ-9 below 10"
    return "can't tell"


def score_audio(audio: np.ndarray) -> dict:
    """Raw scores and levels for one person's speech at 16 kHz."""
    if not _MODEL:
        ckpt = checkpoint_path()
        if ckpt is None:
            raise RuntimeError("The Kintsugi add-on isn't installed.")
        _MODEL.append(_Model(ckpt))
    raw = _MODEL[0](_features(audio))
    out = {}
    for task, score in raw.items():
        level = _level(task, score)
        out[task] = {"score": round(score, 4), "level": level, "label": LEVELS[task][level]}
    out["depression"]["screen"] = screen(raw["depression"])
    return out


def participant_audio(wav_path: str, segments: list[dict], speaker: str) -> np.ndarray:
    """The participant's own speech, joined, from the processing copy of the audio."""
    import soundfile as sf

    pieces = []
    with sf.SoundFile(wav_path) as f:
        for seg in segments:
            if seg.get("speaker") != speaker:
                continue
            a, b = int(seg["start"] * SAMPLE_RATE), int(seg["end"] * SAMPLE_RATE)
            if b <= a:
                continue
            f.seek(a)
            pieces.append(f.read(b - a, dtype="float32"))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def run(wav_path: str, segments: list[dict], speaker: Optional[str]) -> Optional[dict]:
    """Kintsugi estimates for the participant, or a note saying why not."""
    if not available():
        return None
    if not speaker:
        return {"skipped": "No single participant to analyse (group, or roles not confirmed)."}
    audio = participant_audio(wav_path, segments, speaker)
    seconds = len(audio) / SAMPLE_RATE
    if seconds < MIN_SPEECH_S:
        return {"skipped": f"Only {seconds:.0f} s of participant speech; the model needs "
                           f"{MIN_SPEECH_S:.0f} s."}
    result = score_audio(audio)
    result.update({
        "speech_s": round(seconds, 1),
        "model": "KintsugiHealth/dam 3.1 (Apache-2.0)",
        "note": ("Voice-only research estimate from Kintsugi's open model. Trained on "
                 "their data; not validated on this population or on clinical interviews; "
                 "not a diagnosis."),
    })
    return result


def per_speaker(wav_path: str, segments: list[dict]) -> Optional[dict]:
    """Estimates for every speaker with enough speech (roles are decided later)."""
    if not available():
        return None
    talk: dict[str, float] = {}
    for seg in segments:
        talk[seg.get("speaker")] = talk.get(seg.get("speaker"), 0.0) + max(
            0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
    out = {}
    for spk, secs in talk.items():
        if secs < MIN_SPEECH_S:
            continue
        audio = participant_audio(wav_path, segments, spk)
        out[spk] = {**score_audio(audio), "speech_s": round(len(audio) / SAMPLE_RATE, 1)}
    return out


def for_subject(per: Optional[dict], subject: Optional[str]) -> Optional[dict]:
    """The participant's estimate from :func:`per_speaker`, or why there is none."""
    if per is None:
        return None
    base = {"model": "KintsugiHealth/dam 3.1 (Apache-2.0)", "note": (
        "Voice-only research estimate from Kintsugi's open model. Trained on their data; "
        "not validated on this population or on clinical interviews; not a diagnosis. "
        + PERFORMANCE_NOTE)}
    if not subject:
        return {**base, "skipped": "No single participant (a group, or roles not confirmed)."}
    if subject not in per:
        return {**base, "skipped": f"The participant has under {MIN_SPEECH_S:.0f} s of speech."}
    return {**base, **per[subject]}
