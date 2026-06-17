#!/usr/bin/env python3
"""
LLM Embeddings Module for ClinicalWhisper
Extracts hidden-state embeddings from HuggingFace language models for use as
deep feature vectors alongside eGeMAPS acoustic features.

Researchers can feed these embeddings into their own ML classifiers
(SVM, logistic regression, neural nets) for clinical speech analysis.

Config (config.yaml):
    embeddings:
      enabled: false
      model: 'sentence-transformers/all-MiniLM-L6-v2'
      pooling: 'mean'   # or 'cls'
      export_format: 'csv'
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger("ClinicalWhisper.embeddings")

# ──────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_POOLING = "mean"  # "mean" or "cls"
DEFAULT_EXPORT_FORMAT = "csv"
BATCH_SIZE = 32  # segments processed per forward pass to limit RAM


# ──────────────────────────────────────────────────────────────────────
# Module-level model cache
# ──────────────────────────────────────────────────────────────────────
_model_cache: dict[str, tuple[Any, Any]] = {}


def _get_model(model_name: str | None = None) -> tuple[Any, Any]:
    """Lazily load and cache a HuggingFace tokenizer + model pair.

    Models are loaded in eval mode on CPU with no gradient tracking.
    The first call for a given model_name triggers a download if the
    model is not already cached on disk.

    Args:
        model_name: HuggingFace repo id. Defaults to ``DEFAULT_MODEL``.

    Returns:
        (tokenizer, model) tuple.
    """
    name = model_name or DEFAULT_MODEL
    if name in _model_cache:
        return _model_cache[name]

    log.info("Loading embedding model '%s' (first use may download weights)...", name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        with torch.no_grad():
            model = AutoModel.from_pretrained(name)
        model.eval()
        log.info("Embedding model '%s' loaded (hidden_size=%d)", name, model.config.hidden_size)
    except Exception as exc:
        log.error("Failed to load embedding model '%s': %s", name, exc)
        raise

    _model_cache[name] = (tokenizer, model)
    return tokenizer, model


# ──────────────────────────────────────────────────────────────────────
# Internal pooling helpers
# ──────────────────────────────────────────────────────────────────────

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, respecting the attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """Extract the [CLS] token embedding (first token)."""
    return last_hidden[:, 0, :]


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def extract_text_embeddings(
    text: str,
    model_name: str | None = None,
    pooling: str = DEFAULT_POOLING,
) -> np.ndarray:
    """Extract a single embedding vector for the given text.

    Args:
        text:       Input text to embed.
        model_name: HuggingFace repo id (or None for default).
        pooling:    ``'mean'`` for mean-pooling of the last hidden layer,
                    or ``'cls'`` for the [CLS] token hidden state.

    Returns:
        1-D numpy array of shape ``(hidden_dim,)``.
    """
    if not text or not text.strip():
        tokenizer, model = _get_model(model_name)
        dim = model.config.hidden_size
        log.warning("Empty text provided; returning zero vector (dim=%d)", dim)
        return np.zeros(dim, dtype=np.float32)

    tokenizer, model = _get_model(model_name)
    max_len = getattr(tokenizer, "model_max_length", 512)
    # Cap to a sane upper bound (some tokenizers report 1e30)
    if max_len > 8192:
        max_len = 512

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Warn if truncation occurred
    token_count = encoded["input_ids"].shape[1]
    if token_count >= max_len:
        log.warning(
            "Input text (%d chars) was truncated to %d tokens (model max_length=%d)",
            len(text), max_len, max_len,
        )

    with torch.no_grad():
        outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    if pooling == "cls":
        embedding = _cls_pool(last_hidden)
    else:
        embedding = _mean_pool(last_hidden, encoded["attention_mask"])

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def extract_segment_embeddings(
    segments: list[dict],
    model_name: str | None = None,
    pooling: str = DEFAULT_POOLING,
) -> np.ndarray:
    """Extract embeddings for every segment in a ClinicalWhisper transcript.

    Processes segments in batches to keep memory usage bounded.

    Args:
        segments:   List of segment dicts, each with at least a ``"text"`` key.
        model_name: HuggingFace repo id (or None for default).
        pooling:    ``'mean'`` or ``'cls'``.

    Returns:
        2-D numpy array of shape ``(num_segments, hidden_dim)``.
    """
    if not segments:
        log.warning("No segments provided; returning empty array")
        return np.empty((0, 0), dtype=np.float32)

    tokenizer, model = _get_model(model_name)
    hidden_dim = model.config.hidden_size
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len > 8192:
        max_len = 512

    total = len(segments)
    all_embeddings = np.empty((total, hidden_dim), dtype=np.float32)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = [
            (seg.get("text") or "").strip() or "[empty]"
            for seg in segments[batch_start:batch_end]
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded)

        last_hidden = outputs.last_hidden_state

        if pooling == "cls":
            batch_emb = _cls_pool(last_hidden)
        else:
            batch_emb = _mean_pool(last_hidden, encoded["attention_mask"])

        all_embeddings[batch_start:batch_end] = batch_emb.cpu().numpy()

        if total > BATCH_SIZE:
            log.info(
                "Embedding progress: %d / %d segments",
                min(batch_end, total), total,
            )

    log.info("Extracted embeddings for %d segments (dim=%d)", total, hidden_dim)
    return all_embeddings


def extract_speaker_embeddings(
    segments: list[dict],
    roles: dict[str, str] | None = None,
    model_name: str | None = None,
    pooling: str = DEFAULT_POOLING,
) -> dict[str, np.ndarray]:
    """Extract one embedding per speaker by concatenating their utterances.

    Args:
        segments:   ClinicalWhisper segment list (each has ``"speaker"`` and ``"text"``).
        roles:      Optional mapping from raw speaker labels to role names,
                    e.g. ``{"Speaker 1": "clinician", "Speaker 2": "patient"}``.
                    If None, the raw ``"speaker"`` label is used as the key.
        model_name: HuggingFace repo id (or None for default).
        pooling:    ``'mean'`` or ``'cls'``.

    Returns:
        Dict mapping role (or speaker label) to a 1-D numpy array of
        shape ``(hidden_dim,)``.
    """
    if not segments:
        log.warning("No segments provided for speaker embeddings")
        return {}

    roles = roles or {}

    # Gather text per speaker
    speaker_texts: dict[str, list[str]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "Speaker 1")
        role = roles.get(speaker, speaker)
        text = (seg.get("text") or "").strip()
        if text:
            speaker_texts.setdefault(role, []).append(text)

    result: dict[str, np.ndarray] = {}
    for role, texts in speaker_texts.items():
        combined = " ".join(texts)
        log.info(
            "Extracting embedding for '%s' (%d utterances, %d chars)",
            role, len(texts), len(combined),
        )
        result[role] = extract_text_embeddings(combined, model_name=model_name, pooling=pooling)

    return result


# ──────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────

def export_embeddings(
    embeddings: np.ndarray,
    output_path: str,
    format: str = DEFAULT_EXPORT_FORMAT,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Export embedding array to disk.

    Args:
        embeddings:  1-D or 2-D numpy array.
        output_path: Destination file path (extension is added if missing).
        format:      ``'csv'`` or ``'npy'``.
        metadata:    Optional metadata dict written as a comment header in CSV.

    Returns:
        Absolute path of the written file.
    """
    # Ensure 2-D for uniform handling
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    fmt = format.lower().strip(".")

    if fmt == "npy":
        if dest.suffix != ".npy":
            dest = dest.with_suffix(".npy")
        np.save(str(dest), embeddings)
        log.info("Saved embeddings to %s (shape %s)", dest, embeddings.shape)
        return str(dest)

    # Default: CSV
    if dest.suffix != ".csv":
        dest = dest.with_suffix(".csv")

    num_dims = embeddings.shape[1]
    header = [f"dim_{i}" for i in range(num_dims)]

    with open(dest, "w", newline="", encoding="utf-8") as fh:
        # Metadata comment block
        fh.write(f"# ClinicalWhisper LLM Embeddings\n")
        fh.write(f"# rows={embeddings.shape[0]}, dims={num_dims}\n")
        if metadata:
            for key, value in metadata.items():
                fh.write(f"# {key}={value}\n")

        writer = csv.writer(fh)
        writer.writerow(header)
        for row in embeddings:
            writer.writerow([f"{v:.8f}" for v in row])

    log.info("Saved embeddings CSV to %s (shape %s)", dest, embeddings.shape)
    return str(dest)


# ──────────────────────────────────────────────────────────────────────
# Memory management
# ──────────────────────────────────────────────────────────────────────

def unload_model(model_name: str | None = None) -> None:
    """Remove a model from the cache and free memory.

    Args:
        model_name: Model to unload.  If None, **all** cached models are freed.
    """
    if model_name is None:
        names = list(_model_cache.keys())
        for name in names:
            del _model_cache[name]
        log.info("Unloaded all cached embedding models (%d total)", len(names))
    elif model_name in _model_cache:
        del _model_cache[model_name]
        log.info("Unloaded embedding model '%s'", model_name)
    else:
        log.debug("Model '%s' not in cache; nothing to unload", model_name)

    gc.collect()

    # Free Apple Silicon unified memory if available
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """CLI: extract embeddings from a ClinicalWhisper analysis JSON."""
    parser = argparse.ArgumentParser(
        prog="llm_embeddings",
        description=(
            "Extract LLM hidden-state embeddings from a ClinicalWhisper "
            "analysis JSON and export for downstream ML classifiers."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a *_analysis.json file produced by ClinicalWhisper.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: <input_stem>_embeddings.<format>).",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--pooling", "-p",
        choices=["mean", "cls"],
        default=DEFAULT_POOLING,
        help=f"Pooling strategy (default: {DEFAULT_POOLING}).",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "npy"],
        default=DEFAULT_EXPORT_FORMAT,
        help=f"Export format (default: {DEFAULT_EXPORT_FORMAT}).",
    )
    parser.add_argument(
        "--speaker",
        action="store_true",
        help="Export per-speaker embeddings instead of per-segment.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # Load analysis JSON
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to parse JSON from %s: %s", input_path, exc)
        sys.exit(1)

    segments = data.get("segments", [])
    if not segments:
        log.error("No segments found in %s", input_path)
        sys.exit(1)

    log.info("Loaded %d segments from %s", len(segments), input_path.name)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        suffix = "_speaker_embeddings" if args.speaker else "_embeddings"
        out_path = str(input_path.with_name(input_path.stem + suffix))

    # Extract and export
    metadata = {
        "source": input_path.name,
        "model": args.model,
        "pooling": args.pooling,
    }

    if args.speaker:
        speaker_embs = extract_speaker_embeddings(
            segments, model_name=args.model, pooling=args.pooling,
        )
        if not speaker_embs:
            log.error("No speaker embeddings extracted")
            sys.exit(1)

        # Stack into matrix with speaker labels in metadata
        speakers = sorted(speaker_embs.keys())
        matrix = np.stack([speaker_embs[s] for s in speakers])
        metadata["speakers"] = ";".join(speakers)
        result_path = export_embeddings(matrix, out_path, format=args.format, metadata=metadata)
    else:
        embeddings = extract_segment_embeddings(
            segments, model_name=args.model, pooling=args.pooling,
        )
        metadata["segments"] = str(len(segments))
        result_path = export_embeddings(embeddings, out_path, format=args.format, metadata=metadata)

    print(f"Embeddings exported to: {result_path}")

    # Clean up
    unload_model()


if __name__ == "__main__":
    main()
