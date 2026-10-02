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

# Masking for languages other than English (OpenMed's multilingual privacy
# filter, MLX build, Apache-2.0, 2.8 GB). Kept in OpenMed's own cache layout
# (``openmed/<org>_<name>/weights.safetensors``), which is how OpenMed finds
# converted MLX weights.
LANGUAGES_ROOT = bundled_models.APP_SUPPORT / "addons" / "languages"
LANGUAGES_MODEL = "OpenMed/privacy-filter-multilingual-mlx"
# What that model was trained to mask (its model card).
MASKABLE = {"ar", "bn", "zh", "nl", "en", "fr", "de", "hi", "it", "ja", "ko", "pt",
            "es", "te", "tr", "vi"}


def _openmed_dir(root: Path, model: str = LANGUAGES_MODEL) -> Path:
    return root / "openmed" / model.replace("/", "_")


def languages_cache(root: Optional[Path] = None) -> Optional[Path]:
    """OpenMed cache folder holding the multilingual masker, if installed."""
    root = root or LANGUAGES_ROOT
    d = _openmed_dir(root)
    if (d / "weights.safetensors").is_file() and (d / "config.json").is_file():
        return root / "openmed"
    return None


def languages_available() -> bool:
    return languages_cache() is not None


def install_languages(source: Path, root: Optional[Path] = None) -> Path:
    """Copy the multilingual masker from the add-on folder the user picked."""
    root = root or LANGUAGES_ROOT
    name = LANGUAGES_MODEL.replace("/", "_")
    found = None
    for base in [source, source / "openmed"] + (
            [p / "openmed" for p in source.iterdir() if p.is_dir()] if source.is_dir() else []):
        if (base / name / "weights.safetensors").is_file():
            found = base / name
            break
    if found is None:
        raise ValueError("That folder doesn't contain the ClinicalWhisper Languages add-on.")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".installing-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(found, tmp / "openmed" / name)
        target = root / "openmed"
        if target.exists():
            shutil.rmtree(target)
        (tmp / "openmed").rename(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("Languages add-on installed.")
    return root / "openmed"


def install_kintsugi(source: Path, root: Optional[Path] = None) -> Path:
    """Copy Kintsugi's checkpoint from the add-on folder the user picked."""
    import kintsugi_dam

    root = root or kintsugi_dam.ROOT
    hits = [p for p in [source / kintsugi_dam.CHECKPOINT, *Path(source).glob(f"*/{kintsugi_dam.CHECKPOINT}")]
            if p.is_file()]
    if not hits:
        raise ValueError("That folder doesn't contain the Kintsugi voice model add-on.")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".installing-{uuid.uuid4().hex[:8]}"
    shutil.copy2(hits[0], tmp)
    tmp.replace(root / kintsugi_dam.CHECKPOINT)
    log.info("Kintsugi voice model add-on installed.")
    return root / kintsugi_dam.CHECKPOINT


PRAAT_ROOT = bundled_models.APP_SUPPORT / "addons" / "praat"


def add_praat_path(root: Optional[Path] = None) -> None:
    """Make the Praat add-on importable (it lives outside the app; GPL-3.0)."""
    site = (root or PRAAT_ROOT) / "site"
    if (site / "parselmouth").exists() or any(site.glob("parselmouth*.so")):
        if str(site) not in sys.path:
            sys.path.append(str(site))


def install_praat(source: Path, root: Optional[Path] = None) -> Path:
    """Copy the Praat add-on's ``site`` folder into Application Support."""
    root = root or PRAAT_ROOT
    candidates = [source / "site", *[p / "site" for p in Path(source).iterdir() if p.is_dir()]]
    site = next((c for c in candidates if any(c.glob("parselmouth*"))), None)
    if site is None:
        raise ValueError("That folder doesn't contain the Praat add-on.")
    tmp = root.parent / f".installing-praat-{uuid.uuid4().hex[:8]}"
    shutil.copytree(site, tmp / "site")
    for extra in ("LICENSE-GPL3.txt", "SOURCE.txt"):
        if (site.parent / extra).is_file():
            shutil.copy2(site.parent / extra, tmp / extra)
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    log.info("Praat add-on installed.")
    return root


def install_deid(source: Path, root: Optional[Path] = None) -> Path:
    """Copy the word aligner (audio de-identification add-on) into place."""
    import audio_deid

    root = root or audio_deid.ROOT
    name = "models--" + audio_deid.MODEL.replace("/", "--")
    candidates = [source / "hub", *[p / "hub" for p in Path(source).iterdir() if p.is_dir()]]
    hub = next((c for c in candidates if (c / name).is_dir()), None)
    if hub is None:
        raise ValueError("That folder doesn't contain the audio de-identification add-on.")
    tmp = root.parent / f".installing-deid-{uuid.uuid4().hex[:8]}"
    shutil.copytree(hub / name, tmp / "hub" / name)
    if root.exists():
        shutil.rmtree(root)
    tmp.rename(root)
    log.info("Audio de-identification add-on installed.")
    return root


def install_any(source: Path) -> str:
    """Install whichever add-on ``source`` holds: "scoring", "kintsugi" or "languages"."""
    import kintsugi_dam

    source = Path(source)
    if find_in(source) is not None:
        install(source)
        return "scoring"
    import audio_deid
    deid_name = "models--" + audio_deid.MODEL.replace("/", "--")
    if (source / "hub" / deid_name).is_dir() or any(source.glob(f"*/hub/{deid_name}")):
        install_deid(source)
        return "deid"
    if any(source.glob("site/parselmouth*")) or any(source.glob("*/site/parselmouth*")):
        install_praat(source)
        return "praat"
    if (source / kintsugi_dam.CHECKPOINT).is_file() or any(source.glob(f"*/{kintsugi_dam.CHECKPOINT}")):
        install_kintsugi(source)
        return "kintsugi"
    install_languages(source)
    return "languages"
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
