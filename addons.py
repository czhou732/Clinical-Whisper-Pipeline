"""Clinical scoring as an installable add-on.

The scoring model (Llama-3 8B, 4-bit, 5.3 GB) is more than half of a full
build, and many users only need transcripts, masking and voice measures. So the
base app ships without it, and the model comes as a separate download: a folder
named "ClinicalWhisper Scoring" holding a Hugging Face cache layout
(``hub/models--org--name/snapshots/<rev>/``).

Installing copies that folder's model into Application Support, outside the
read-only app bundle, so it survives app updates. Lookup order for the scorer:

1. the installed add-on;
2. the app bundle (a full build that still carries the model);
3. the model name as given, which on a source checkout resolves through the
   normal Hugging Face cache.
"""

from __future__ import annotations

import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

import bundled_models

log = logging.getLogger("ClinicalWhisper")

ADDON_ROOT = bundled_models.APP_SUPPORT / "addons" / "scoring"
SCORING_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
# Files a usable MLX model folder must contain.
_REQUIRED = ("config.json", "tokenizer.json")


def _cache_name(model: str) -> str:
    return "models--" + model.replace("/", "--")


def _snapshot(hub: Path, model: str) -> Optional[Path]:
    """The snapshot folder ``refs/main`` points at, if it holds a usable model."""
    base = hub / _cache_name(model)
    try:
        rev = (base / "refs" / "main").read_text().strip()
    except OSError:
        return None
    snap = base / "snapshots" / rev
    if not all((snap / f).is_file() for f in _REQUIRED):
        return None
    if not any(snap.glob("*.safetensors")):
        return None
    return snap


def installed_path(model: str = SCORING_MODEL, root: Path = ADDON_ROOT) -> Optional[Path]:
    """Folder of the installed add-on model, or None."""
    return _snapshot(root / "hub", model)


def resolve(model: str) -> str:
    """What to pass to ``mlx_lm.load``: a local folder when one exists."""
    for hub in (ADDON_ROOT / "hub", _bundle_hub()):
        if hub is None:
            continue
        snap = _snapshot(hub, model)
        if snap is not None:
            return str(snap)
    return model


def _bundle_hub() -> Optional[Path]:
    models = bundled_models.bundle_root()
    return models / "hub" if models else None


def scoring_available(model: str = SCORING_MODEL) -> bool:
    """Whether clinical scoring can run on this Mac without a download.

    From a source checkout the model can always be fetched, so this is True.
    """
    if not getattr(sys, "frozen", False):
        return True
    return resolve(model) != model


def find_in(source: Path, model: str = SCORING_MODEL) -> Optional[Path]:
    """Locate the add-on's ``hub`` folder inside what the user picked.

    Accepts the mounted add-on volume, the "ClinicalWhisper Scoring" folder in
    it, or the ``hub`` folder itself.
    """
    candidates = [source, source / "hub"]
    candidates += [p / "hub" for p in source.iterdir() if p.is_dir()] if source.is_dir() else []
    for hub in candidates:
        if hub.name == "hub" and _snapshot(hub, model) is not None:
            return hub
    return None


def install(source: Path, model: str = SCORING_MODEL, root: Path = ADDON_ROOT) -> Path:
    """Copy the add-on model from ``source`` and return the installed folder.

    Copies into a temporary folder first and renames it into place, so an
    interrupted copy never leaves a half-installed model that looks usable.
    Raises ``ValueError`` when ``source`` holds no add-on.
    """
    hub = find_in(Path(source), model)
    if hub is None:
        raise ValueError("That folder doesn't contain the ClinicalWhisper Scoring add-on.")
    name = _cache_name(model)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".installing-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(hub / name, tmp / "hub" / name, symlinks=False)
        target = root / "hub"
        old = None
        if target.exists():
            old = root / f".old-{uuid.uuid4().hex[:8]}"
            target.rename(old)
        (tmp / "hub").rename(target)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    path = installed_path(model, root)
    if path is None:  # pragma: no cover - copy succeeded but layout is wrong
        raise ValueError("The add-on copied, but its model files are incomplete.")
    log.info("Clinical scoring add-on installed.")
    return path
