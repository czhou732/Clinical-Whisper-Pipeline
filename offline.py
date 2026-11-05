"""Keep ClinicalWhisper off the network while it processes recordings.

Two layers:

1. **Hugging Face offline mode.** ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
   make every model loader use local files only. They are read when
   ``huggingface_hub`` and ``transformers`` are imported, so :func:`lock` must
   run before those imports.
2. **A connection guard.** Outgoing connections from this process to anything
   but this machine (loopback) are refused and logged. The first layer covers
   the libraries we know about; the guard covers the ones we do not, so
   "offline" does not depend on every dependency behaving.

Models are downloaded once, on purpose, with ``clinicalwhisper-setup`` (see
:func:`main`); after that nothing needs the network. Set
``CLINICALWHISPER_ALLOW_NETWORK=1`` to opt out, e.g. for the Plaud integration,
which uploads by design.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import platform
import socket
import sys
from typing import Optional

log = logging.getLogger("ClinicalWhisper.offline")

ALLOW_ENV = "CLINICALWHISPER_ALLOW_NETWORK"

# (repository, file or None for the whole snapshot). Only what a run needs;
# the scoring model is listed separately because --transcribe-only skips it.
TRANSCRIPTION_MODELS: list[tuple[str, Optional[str]]] = [
    ("OpenMOSS-Team/MOSS-Transcribe-Diarize", None),
    ("OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1", None),
    ("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx"),
]
# Apple Silicon scores with the MLX build; other machines with the HF weights
# (llm_scoring.mlx_model / hf_model in the config).
SCORING_MODELS: list[tuple[str, Optional[str]]] = [
    ("mlx-community/Meta-Llama-3-8B-Instruct-4bit", None)
    if sys.platform == "darwin" and platform.machine() == "arm64"
    else ("NousResearch/Meta-Llama-3-8B-Instruct", None),
]

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_locked = False


def network_allowed() -> bool:
    return os.environ.get(ALLOW_ENV, "").strip().lower() in ("1", "true", "yes")


def _is_local(address) -> bool:
    """Loopback and Unix sockets are this machine; everything else is not."""
    if not isinstance(address, tuple):
        return True  # AF_UNIX path
    host = str(address[0])
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False  # an unresolved hostname: not provably local


def _guarded(original):
    def connect(sock, address):
        if not _is_local(address):
            log.warning("Blocked a network connection to %s (offline mode).", address[0])
            raise ConnectionRefusedError(
                f"ClinicalWhisper is offline: connection to {address[0]} refused. "
                f"Set {ALLOW_ENV}=1 to allow network access."
            )
        return original(sock, address)
    return connect


def lock() -> bool:
    """Switch this process to offline mode. Returns True if the lock is on.

    Call it first thing in every entry point, before anything imports
    ``huggingface_hub`` or ``transformers``.
    """
    global _locked
    if network_allowed():
        log.info("Network access allowed (%s=1).", ALLOW_ENV)
        return False
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    if not _locked:
        socket.socket.connect = _guarded(_original_connect)
        socket.socket.connect_ex = _guarded(_original_connect_ex)
        _locked = True
    return True


def unlock() -> None:
    """Remove the connection guard (setup downloads and tests only)."""
    global _locked
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    _locked = False


def _present(repo: str, filename: Optional[str]) -> bool:
    """True when a model's config and weights are on disk.

    Whole-snapshot checks are too strict: a cache missing only README or
    LICENSE files loads fine but fails ``snapshot_download(local_files_only=True)``.
    """
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename or "config.json")
    if not isinstance(path, str):
        return False
    if filename:
        return True
    snapshot = Path(path).parent
    if repo.startswith("OpenMed/"):
        # On Apple Silicon OpenMED runs a converted copy kept in its own cache;
        # the Hugging Face snapshot then holds only the tokenizer and config.
        converted = Path.home() / ".cache" / "openmed" / repo.replace("/", "_")
        if (converted / "weights.safetensors").exists():
            return True
    return any(next(snapshot.rglob(pattern), None) is not None for pattern in
               ("*.safetensors", "*.bin", "*.npz", "*.onnx", "*.gguf"))


def missing_models(scoring: bool = True) -> list[str]:
    """Models a run needs that are not on disk yet."""
    try:
        import bundled_models

        if bundled_models.bundle_root() is not None:
            return []  # the packaged app carries its own weights
    except ImportError:
        pass
    wanted = TRANSCRIPTION_MODELS + (SCORING_MODELS if scoring else [])
    return [repo for repo, name in wanted if not _present(repo, name)]


def require_models(scoring: bool = True) -> None:
    """Stop with instructions, rather than a loader error, if models are missing."""
    missing = missing_models(scoring)
    if missing:
        raise SystemExit(
            "ClinicalWhisper runs offline and these models are not downloaded yet:\n  "
            + "\n  ".join(missing)
            + "\nRun `clinicalwhisper-setup` once (this is the only step that uses the "
            "network), then try again."
        )


def main(argv: Optional[list[str]] = None) -> None:
    """``clinicalwhisper-setup``: download every model once, then verify offline."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="clinicalwhisper-setup",
        description="Download the models ClinicalWhisper needs. The only step that "
                    "uses the network; every later run is offline.",
    )
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Skip the clinical scoring model (~4.5 GB).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    unlock()
    from huggingface_hub import hf_hub_download, snapshot_download

    wanted = TRANSCRIPTION_MODELS + ([] if args.transcribe_only else SCORING_MODELS)
    for repo, filename in wanted:
        log.info("Downloading %s ...", repo)
        if filename:
            hf_hub_download(repo, filename)
        else:
            snapshot_download(repo)

    missing = [repo for repo, name in wanted if not _present(repo, name)]
    if missing:
        sys.exit("Still missing after download: " + ", ".join(missing))
    # OpenMED converts its model for Apple Silicon on first use; do that now,
    # while the network is allowed, rather than on the first real recording.
    from pii_scrubber import PIIScrubber

    PIIScrubber().scrub_text("Setup check for Jane Doe.")
    log.info("All models are on disk. ClinicalWhisper will now run without the network.")


if __name__ == "__main__":
    main()
