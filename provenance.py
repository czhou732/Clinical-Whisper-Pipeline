"""Record exactly what produced an analysis, so a result can be reproduced.

A score is only citable if you can say which model weights and which settings
generated it. The analysis JSON previously carried a model *name* and
``pipeline_version: 5.0`` — not enough to reconstruct a run months later, when
a model repo may have been updated in place.

Everything here is best-effort: provenance must never be the reason a job fails.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("ClinicalWhisper")

from version import __version__ as APP_VERSION  # noqa: E402


def _hf_revision(repo_id: str) -> Optional[str]:
    """Resolved commit hash for a model in the local HF cache."""
    try:
        cache = os.environ.get("HF_HUB_CACHE") or str(
            Path.home() / ".cache" / "huggingface" / "hub"
        )
        folder = Path(cache) / ("models--" + repo_id.replace("/", "--"))
        main_ref = folder / "refs" / "main"
        if main_ref.exists():
            return main_ref.read_text(encoding="utf-8").strip()
        snapshots = folder / "snapshots"
        if snapshots.is_dir():
            entries = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
            if entries:
                return entries[0]
    except Exception as exc:  # pragma: no cover - provenance is best-effort
        log.debug("Could not resolve revision for %s: %s", repo_id, exc)
    return None


def _git_commit() -> Optional[str]:
    """Source commit, when running from a checkout rather than a bundle."""
    if getattr(sys, "frozen", False):
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover
        pass
    return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "mlx", "mlx_lm", "openmed", "opensmile", "av"):
        try:
            module = __import__(name)
            v = getattr(module, "__version__", None)
            if v:
                versions[name] = str(v)
        except Exception:  # pragma: no cover
            continue
    return versions


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble the provenance block for one analysis."""
    moss_cfg = cfg.get("moss", {}) or {}
    llm_cfg = cfg.get("llm_scoring", {}) or {}
    pii_cfg = cfg.get("pii_scrubbing", {}) or {}

    moss_repo = moss_cfg.get("model", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
    llm_repo = llm_cfg.get("mlx_model", "mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    pii_repo = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

    record: dict[str, Any] = {
        "app_version": APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "models": {
            "transcription": {"repo": moss_repo, "revision": _hf_revision(moss_repo),
                              "dtype": moss_cfg.get("dtype", "auto"),
                              "device": moss_cfg.get("device", "auto")},
            "clinical_scoring": {"repo": llm_repo, "revision": _hf_revision(llm_repo),
                                 "max_tokens": llm_cfg.get("max_tokens"),
                                 "samples": llm_cfg.get("samples", 1),
                                 "temperature": llm_cfg.get("temperature", 0.0),
                                 "transcript_scope": llm_cfg.get("transcript_scope", "dialogue")},
            "deidentification": {"repo": pii_repo, "revision": _hf_revision(pii_repo),
                                 "confidence_threshold": pii_cfg.get("confidence_threshold", 0.7)},
        },
        "packages": _package_versions(),
    }

    commit = _git_commit()
    if commit:
        record["source_commit"] = commit

    return record
