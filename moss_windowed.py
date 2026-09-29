"""Windowed, batched MOSS transcription on MLX.

Long recordings are split into equal-length overlapping windows (see
``moss_chunking``), decoded together in memory-sized batches on MLX, stitched
back into one transcript, and speaker labels are linked across windows by
voice. This is the Apple Silicon path of :class:`moss_diarizer.MOSSDiarizer`;
``d`` below is that diarizer, which owns the loaded model and settings.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

try:
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        load_audio_item,
    )
except ImportError:  # pragma: no cover - only without the diarization extra
    parse_transcript = build_transcription_messages = load_audio_item = None

from moss_chunking import (
    UNCERTAIN_TALK_S,
    consolidate_speakers,
    flag_uncertain_speakers,
    is_silent,
    limit_speakers,
    link_speakers,
    looks_degenerate,
    merge_windows,
    plan_windows,
    speaker_embeddings,
)

log = logging.getLogger("ClinicalWhisper")


# Memory the batched decoder's key/value cache may use: at most a quarter of
# physical RAM, and at most 60% of what is free right now. A batch that spills
# into swap ran ~15x slower in testing, and an app killed by macOS for memory
# vanishes without an error message.
_KV_TOTAL_FRACTION = 0.25
_KV_AVAILABLE_FRACTION = 0.6
# Measured ~8 output tokens per second of conversational audio; the KV cache is
# pre-sized for this plus headroom rather than the worst-case token budget.
_EXPECTED_TOKENS_PER_SECOND = 9.5


def _memory_bytes() -> tuple[float, Optional[float]]:
    """Physical RAM, and memory available without swapping (macOS only).

    Uses the kernel's own availability level — the number behind
    ``memory_pressure``'s "memory free percentage", which counts memory the
    OS can reclaim by compressing idle apps. Summing ``vm_stat``'s free and
    inactive pages instead read ~8 GB on a machine macOS called 45% free
    (~15 GB), and halved the batch for no reason.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = 16e9
    try:
        import subprocess

        level = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return total, total * int(level) / 100
    except (OSError, ValueError, subprocess.SubprocessError):
        return total, None


# Cross-window speaker thresholds, per embedding type, tuned on AMI human labels
# (evals/ground_truth/tune_linking.py; EN2002a-c). Voice-model embeddings put
# the same person across halves of a meeting at 0.75-0.93 and different people
# at most 0.57. Results formed a broad plateau — cpWER 0.298, DER 0.208 for link
# 0.50, any merge value, split 0.75-0.85 — so these sit mid-plateau.
# SPLIT lets two labels in one window join when the voices match this closely:
# MOSS sometimes gives one person two labels (5-7 labels in 4-person windows),
# and forbidding that merge alone cost ~7 points of cpWER.
_VOICE_LINK, _VOICE_MERGE, _VOICE_SPLIT = 0.50, 0.65, 0.75
_ENCODER_LINK, _ENCODER_MERGE = 0.50, 0.75

_VOICE: list = []


def _voice_embedder():
    """Load the speaker model once; None if unavailable (fall back to encoder)."""
    if not _VOICE:
        try:
            from voice_embedder import VoiceEmbedder

            _VOICE.append(VoiceEmbedder.load())
        except ImportError:
            _VOICE.append(None)
    return _VOICE[0]


def _dump_windows(audio_path, windows, window_segments, embeddings) -> None:
    """With CW_DUMP_WINDOWS=<dir>, save per-window output for offline tuning."""
    out = os.environ.get("CW_DUMP_WINDOWS")
    if not out:
        return
    import json
    from pathlib import Path

    target = Path(out) / (Path(audio_path).stem + ".windows.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "windows": [[w.index, w.start_sample, w.n_samples, w.sample_rate] for w in windows],
        "segments": window_segments,
        "embeddings": [
            {spk: [vec.tolist(), talk] for spk, (vec, talk) in emb.items()} for emb in embeddings
        ],
    }))



class _WavAudio:
    """A mono WAV at the model's rate, read on demand and sliced like an array.

    A 5-hour interview is ~1.2 GB as float32. Reading each window when it is
    needed keeps memory to the windows in flight, whatever the file length.
    The lock makes one file handle safe for the preparation threads.
    """

    def __init__(self, path: str):
        import soundfile as sf

        self._file = sf.SoundFile(path)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._file.frames

    def __getitem__(self, key: slice) -> np.ndarray:
        start, stop, _ = key.indices(len(self))
        with self._lock:
            self._file.seek(start)
            return self._file.read(max(stop - start, 0), dtype="float32")

    def close(self) -> None:
        self._file.close()


def _open_audio(path: str, sr: int):
    """Stream a mono WAV already at ``sr`` (what the pipeline writes); load
    anything else whole."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate == sr and info.channels == 1:
            return _WavAudio(str(path))
    except (ImportError, RuntimeError):
        pass
    return load_audio_item(str(path), sampling_rate=sr)


def prepare_window(d, audio: np.ndarray) -> dict[str, np.ndarray]:
    """Tokenise the prompt and compute log-mel features for one window (CPU)."""
    if d._window_template is None:
        # The template only marks where audio goes; it never reads it.
        # (MOSS's own prepare_inputs chains fields with ``or``, which
        # fails on an in-memory array, so the processor is called directly.)
        messages = build_transcription_messages("audio", d.prompt)
        d._window_template = d.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    from moss_diarizer import _CONTEXT_LIMIT  # runtime import: avoids a cycle

    inputs = d.processor(
        text=d._window_template,
        audio=[audio],
        max_length=_CONTEXT_LIMIT,
        return_tensors="pt",
    )
    return {
        "input_ids": inputs["input_ids"][0].numpy(),
        "input_features": inputs["input_features"].float().numpy(),
        "audio_feature_lengths": inputs["audio_feature_lengths"].numpy(),
    }


def _text_config(d) -> dict:
    if d.backend == "mlx":
        return d._mlx.text_config
    return d.model.config.text_config.to_dict()


def batch_limit(d, prompt_len: int, expected_new: int) -> int:
    """Largest batch whose key/value cache fits the memory allowance.

    On CUDA the cache lives in GPU memory, so that is what is measured; on
    Apple Silicon memory is unified and the system figure applies.
    """
    cfg = _text_config(d)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    per_token = cfg["num_hidden_layers"] * 2 * cfg["num_key_value_heads"] * head_dim * 2
    per_sequence = (prompt_len + expected_new) * per_token
    if d.backend != "mlx" and d.device.type == "cuda":
        import torch

        free, total = torch.cuda.mem_get_info(d.device)
        available: Optional[float] = float(free)
    else:
        total, available = _memory_bytes()
    allowance = total * _KV_TOTAL_FRACTION
    if available is not None:
        allowance = min(allowance, available * _KV_AVAILABLE_FRACTION)
    return max(1, min(d.batch_size, int(allowance // per_sequence)))

def _decode_torch(d, prepared, budget: int) -> list[list[int]]:
    """Batched greedy decode with PyTorch — the CUDA / non-Apple path.

    Equal-length windows need no padding, so one ``generate`` call decodes the
    whole batch; MOSS's ``audio_chunk_mapping`` routes each 30 s chunk to its
    row.
    """
    import copy

    import torch

    def t(x, dtype=None):
        out = torch.from_numpy(np.ascontiguousarray(x)).to(d.device)
        return out.to(dtype) if dtype is not None else out

    ids = t(np.stack([p["input_ids"] for p in prepared]))
    config = copy.deepcopy(d.model.generation_config)
    config.max_new_tokens = budget
    config.do_sample = False
    with torch.inference_mode():
        out = d.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            input_features=t(np.concatenate([p["input_features"] for p in prepared]), d.dtype),
            audio_feature_lengths=t(np.concatenate([p["audio_feature_lengths"] for p in prepared])),
            audio_chunk_mapping=t(np.concatenate([
                np.full(len(p["audio_feature_lengths"]), i) for i, p in enumerate(prepared)
            ])),
            generation_config=config,
        )
    eos = config.eos_token_id
    stop = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    if config.pad_token_id is not None:
        stop.add(config.pad_token_id)
    rows = []
    for row in out[:, ids.shape[1]:].tolist():
        cut = next((i for i, tok in enumerate(row) if tok in stop), len(row))
        rows.append(row[:cut])
    return rows


def _decode_group(d, group, prepared, sr: int, expected_new: int, on_tokens) -> list[dict]:
    """Encode and decode one batch of equal-length windows (MLX, or PyTorch).

    Returns one ``{"segments", "features", "tokens", "budget"}`` per window,
    with segment times relative to the window's start.
    """
    if d.backend != "mlx":
        budget = d._token_budget(group[0].n_samples / sr, len(prepared[0]["input_ids"]))
        outputs = _decode_torch(d, prepared, budget)
        return [_result(d, tokens, None, budget) for tokens in outputs]

    import mlx.core as mx

    ids = np.stack([p["input_ids"] for p in prepared])
    audio_embeds = d._mlx.audio.encode(
        np.concatenate([p["input_features"] for p in prepared]),
        np.concatenate([p["audio_feature_lengths"] for p in prepared]),
        np.concatenate([
            np.full(len(p["audio_feature_lengths"]), i) for i, p in enumerate(prepared)
        ]),
        len(group),
    )
    embeds = d._mlx.prompt_embeddings(ids, audio_embeds)
    budget = d._token_budget(group[0].n_samples / sr, ids.shape[1])
    outputs = d._mlx.decoder.generate_batch(
        ids, embeds, max_new_tokens=budget,
        eos_token_ids=d._mlx.eos_token_ids,
        expected_new_tokens=expected_new,
        token_callback=on_tokens,
    )
    return [
        _result(d, tokens, np.array(feats.astype(mx.float32)), budget)
        for feats, tokens in zip(audio_embeds, outputs)
    ]


def _result(d, tokens: list[int], features: Optional[np.ndarray], budget: int) -> dict:
    text = d.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "segments": [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": seg.speaker or "UNKNOWN",
                "text": seg.text or "",
            }
            for seg in parse_transcript(text.strip())
        ],
        # Encoder features double as a fallback voice embedding (MLX path only).
        "features": features if features is not None else np.zeros((0, 1)),
        "tokens": len(tokens),
        "budget": budget,
    }


def _redo_in_halves(d, audio: np.ndarray, window, sr: int) -> list[dict]:
    """Re-transcribe a window whose decode went wrong, as two shorter halves.

    A greedy loop or runaway window is usually local; splitting changes the
    context enough to break it. The halves overlap by 15 s and are stitched by
    overlap agreement alone.
    """
    clip = audio[window.start_sample:window.start_sample + window.n_samples]
    halves = plan_windows(len(clip), sr, len(clip) / sr / 2 + 7.5, 15.0)
    prepared = [prepare_window(d, clip[h.start_sample:h.start_sample + h.n_samples]) for h in halves]
    expected = int(halves[0].n_samples / sr * _EXPECTED_TOKENS_PER_SECOND)
    results = _decode_group(d, halves, prepared, sr, expected, None)
    segs = [r["segments"] for r in results]
    empty = [{} for _ in halves]
    maps = link_speakers(halves, segs, empty, similarity_threshold=2.0)  # overlap only
    return merge_windows(halves, segs, maps)


_BUCKET_S = 30  # short files are padded to a multiple of this to share batches


def transcribe_windowed(d, audio_path: str, token_callback) -> list[dict]:
    """Transcribe one file. See :func:`transcribe_many`."""
    return transcribe_many(d, [audio_path], token_callback)[0]


def transcribe_many(d, audio_paths: list[str], token_callback=None) -> list[list[dict]]:
    """Split files into equal windows, decode them in batches, and stitch.

    Windows from *different* files are pooled into the same batches: a folder
    of short interviews otherwise decodes one file at a time, the slowest case
    (batch 1 ran at 142 tok/s against 608 at batch 16). Windows must share a
    length to share a batch, so a file short enough to be one window is padded
    with silence up to a multiple of 30 s; silence yields no text, and segments
    past the real end are dropped. Speakers are linked within each file only.

    See ``moss_chunking`` for why long files are windowed at all: a single pass
    over an hour slows to ~9 tok/s as context grows.
    """
    sr = d.processor.feature_extractor.sampling_rate
    prepare_window(d, np.zeros(sr, dtype=np.float32))  # render the prompt template before threading
    workers = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

    files, units = [], []  # units: (file index, window, padded length)
    for f, path in enumerate(audio_paths):
        audio = _open_audio(path, sr)
        windows = plan_windows(len(audio), sr, d.window_seconds, d.window_overlap_seconds)
        files.append({
            "path": path, "audio": audio, "windows": windows,
            "segments": [[] for _ in windows], "features": [np.zeros((0, 1)) for _ in windows],
        })
        for w in windows:
            clip = audio[w.start_sample:w.start_sample + w.n_samples]
            if is_silent(clip, sr):
                log.info("%s: skipping silent window %d.", path, w.index)
                continue
            bucket = _BUCKET_S * sr
            length = w.n_samples if len(windows) > 1 else -(-w.n_samples // bucket) * bucket
            units.append((f, w, length))

    def _prepare(unit):
        f, w, length = unit
        clip = files[f]["audio"][w.start_sample:w.start_sample + w.n_samples]
        if length > len(clip):
            clip = np.concatenate([clip, np.zeros(length - len(clip), dtype=clip.dtype)])
        return prepare_window(d, clip)

    # Windows are prepared (log-mel, ~15 MB each for 300 s) one round ahead of
    # the decoder rather than all at once: preparing every window of a 5-hour
    # file up front would hold ~1 GB of features before the first decode.
    ordered = sorted(units, key=lambda u: -u[2])  # same order the rounds consume
    pending: dict = {}
    submitted = consumed = 0

    def _prefetch(upto: int) -> None:
        nonlocal submitted
        while submitted < min(upto, len(ordered)):
            u = ordered[submitted]
            pending[id(u)] = workers.submit(_prepare, u)
            submitted += 1

    emitted, rounds, redo = 0, 0, []
    try:
        # Longest windows first, so memory-heavy batches run while it is freest.
        for length in sorted({u[2] for u in units}, reverse=True):
            bucket_units = [u for u in units if u[2] == length]
            window_s = length / sr
            est_prompt = int(window_s * 12.5) + 64
            expected_new = int(window_s * _EXPECTED_TOKENS_PER_SECOND)
            done = 0
            while done < len(bucket_units):
                # Re-check memory every round: other apps come and go.
                remaining = len(bucket_units) - done
                batch = batch_limit(d, est_prompt, expected_new)
                size = math.ceil(remaining / math.ceil(remaining / batch))  # balanced rounds
                group = bucket_units[done:done + size]
                t_round = time.perf_counter()
                _prefetch(consumed + 2 * size)
                consumed += size
                prepared = [pending.pop(id(u)).result() for u in group]

                def _cb(n: int, base: int = emitted) -> None:
                    if token_callback is not None:
                        token_callback(base + n)

                results = _decode_group(d, [u[1] for u in group], prepared, sr, expected_new, _cb)
                rounds += 1
                log.info(
                    "Round %d: %d window(s) of %.0f s, %.1fs (%d tokens, longest %d).",
                    rounds, len(group), window_s, time.perf_counter() - t_round,
                    sum(x["tokens"] for x in results), max(x["tokens"] for x in results),
                )
                for (f, w, _), res in zip(group, results):
                    real_end = w.n_samples / sr
                    files[f]["segments"][w.index] = [
                        {**seg, "end": min(seg["end"], real_end)}
                        for seg in res["segments"] if seg["start"] < real_end
                    ]
                    files[f]["features"][w.index] = res["features"]
                    if looks_degenerate(res["segments"], res["tokens"], res["budget"]):
                        redo.append((f, w))
                emitted += sum(x["tokens"] for x in results)
                done += len(group)
    finally:
        workers.shutdown(wait=False, cancel_futures=True)

    for f, w in redo:
        log.warning("%s: window %d (%.0f-%.0fs) looped or ran out its budget; redoing in halves.",
                    files[f]["path"], w.index, w.start, w.end)
        files[f]["segments"][w.index] = _redo_in_halves(d, files[f]["audio"], w, sr)
        files[f]["features"][w.index] = np.zeros((0, 1))

    out = []
    for info in files:
        if len(info["windows"]) == 1:
            segments = info["segments"][0]
        else:
            segments = _link_and_merge(
                d, info["path"], info["audio"], info["windows"],
                info["segments"], info["features"], rounds,
            )
        segments = _apply_speaker_count(d, info["path"], info["audio"], segments)
        if isinstance(info["audio"], _WavAudio):
            info["audio"].close()
        uncertain = flag_uncertain_speakers(segments)
        if uncertain:
            log.info("%s: flagged %d speaker(s) with under %.0f s of talk as uncertain: %s",
                     info["path"], len(uncertain), UNCERTAIN_TALK_S, ", ".join(uncertain))
        out.append(segments)
    return out


def _apply_speaker_count(d, path, audio, segments: list[dict]) -> list[dict]:
    """Fold extra labels into ``d.num_speakers`` speakers, when it is set."""
    n = getattr(d, "num_speakers", None)
    found = len({seg["speaker"] for seg in segments})
    if not n or found <= n:
        return segments
    voice = _voice_embedder()
    embeddings = voice.speakers(segments, audio) if voice is not None else {}
    log.info("%s: %d speaker labels folded into the %d requested.", path, found, n)
    return limit_speakers(segments, embeddings, n)


def _link_and_merge(d, audio_path, audio, windows, window_segments, window_features, rounds):
    t_link = time.perf_counter()
    voice = _voice_embedder()
    if voice is not None:
        embeddings = [
            voice.speakers(segs, audio[w.start_sample:w.start_sample + w.n_samples])
            for w, segs in zip(windows, window_segments)
        ]
        link_at, merge_at, center, source = (
            d.speaker_similarity or _VOICE_LINK, d.speaker_merge_similarity or _VOICE_MERGE,
            False, "voice model",
        )
        split_at = getattr(d, "speaker_split_similarity", None) or _VOICE_SPLIT
    else:
        embeddings = [
            speaker_embeddings(segs, feats)
            for segs, feats in zip(window_segments, window_features)
        ]
        link_at, merge_at, center, source = (
            d.speaker_similarity or _ENCODER_LINK, d.speaker_merge_similarity or _ENCODER_MERGE,
            True, "encoder features",
        )
        split_at = None  # encoder features are too weak to justify a within-window merge
    _dump_windows(audio_path, windows, window_segments, embeddings)
    mappings = link_speakers(windows, window_segments, embeddings, link_at, center=center,
                             split_threshold=split_at)
    mappings = consolidate_speakers(embeddings, mappings, merge_at, center=center,
                                    split_threshold=split_at)
    log.info(
        "Transcribed %d windows in %d round(s); %d speaker(s) after linking by %s (%.1fs).",
        len(windows), rounds, len({g for m in mappings for g in m.values()}), source,
        time.perf_counter() - t_link,
    )
    return merge_windows(windows, window_segments, mappings)
