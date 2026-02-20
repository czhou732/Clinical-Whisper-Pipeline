#!/usr/bin/env python3
"""Shared configuration loader for ClinicalWhisper services."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "medium.en",
    "input_folder": "./Input",
    "processed_folder": "./Processed",
    "output_folder": "./Output",
    "audio_extensions": [".m4a", ".mp3", ".wav", ".mp4"],
    "sentiment": {
        "enabled": True,
        "sentences_per_segment": 5,
    },
    "statistics": {
        "enabled": True,
    },
    "skip_already_processed": True,
    "diarization": {
        "enabled": False,
        "hf_token": "",
        "min_speakers": 2,
        "max_speakers": 6,
    },
    "pipeline": {
        "queue_db_path": "./clinicalwhisper_jobs.db",
        "secure_storage_folder": "./Input/ingestion",
        "analysis_output_folder": "./Output",
        "poll_interval_seconds": 1.0,
        "max_workers": 2,
    },
}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path_value: Union[str, Path]) -> str:
    """Resolve a config path relative to the project root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    cfg = deepcopy(DEFAULT_CONFIG)

    if cfg_path.exists():
        user_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cfg = _deep_merge(cfg, user_cfg)

    return cfg
