"""Point the model loaders at weights shipped inside the app bundle.

The .app carries the full model set (~7.2 GB) so a new machine needs nothing but
the DMG — no Hugging Face download, no account, no network. The bundle is
read-only, which the loaders tolerate as long as the handful of things that
genuinely need writing live elsewhere:

* ``HF_HUB_CACHE`` -> the bundled, read-only weights.
* ``HF_HOME``      -> a writable Application Support directory. ``trust_remote_code``
                      writes MOSS's model definition here (about 72 KB).
* ``HF_HUB_OFFLINE`` -> never reach for the network, so a missing file fails
                      immediately and loudly instead of silently downloading.

``configure()`` must run before ``huggingface_hub`` or ``transformers`` is
imported anywhere, because both read these variables at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "ClinicalWhisper"

_openmed_cache: Optional[Path] = None
_configured = False


def bundle_root() -> Optional[Path]:
    """Directory holding the bundled models, or None when running from source."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    models = base / "models"
    return models if (models / "hub").is_dir() else None


def openmed_cache_dir() -> Optional[str]:
    """Cache directory for OpenMED, when the bundled copy is in use."""
    return str(_openmed_cache) if _openmed_cache else None


def configure() -> bool:
    """Redirect the model loaders at the bundled weights. Returns True if used.

    A no-op when running from source, where the normal ~/.cache locations and
    ordinary online behaviour are what you want.
    """
    global _openmed_cache, _configured

    if _configured:
        return _openmed_cache is not None

    _configured = True
    models = bundle_root()
    if models is None:
        return False

    # Anything the loaders need to write goes here, never into the bundle.
    hf_home = APP_SUPPORT / "hf"
    try:
        hf_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    os.environ.setdefault("HF_HUB_CACHE", str(models / "hub"))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Nothing here should phone home for telemetry either.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    openmed_dir = models / "openmed"
    if openmed_dir.is_dir():
        _openmed_cache = openmed_dir

    return True
