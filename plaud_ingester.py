#!/usr/bin/env python3
"""
PlaudConnector — Watches for Plaud Note audio recordings and auto-submits
them to the ClinicalWhisper pipeline.

Supports two source modes:
  1. USB mount detection (/Volumes/PLAUD*)
  2. App-export folder watching (any directory)

Supports two ingestion modes:
  1. API mode — POST audio to FastAPI endpoint (default)
  2. Direct mode — copy files into ClinicalWhisper's Input folder
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PlaudConnector] %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PlaudConnector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cw_config import load_config, resolve_path  # noqa: E402


def _load_plaud_config(config_path: Optional[str] = None) -> dict:
    cfg = load_config(config_path)
    plaud = cfg.get("plaud", {})

    defaults = {
        "enabled": True,
        "watch_paths": ["/Volumes/PLAUD*", "~/Downloads/PlaudExports"],
        "ingestion_mode": "api",
        "api_url": "http://127.0.0.1:8000/jobs",
        "audio_extensions": [".mp3", ".wav", ".m4a"],
        "ingest_transcriptions": False,
        "transcript_extensions": [".txt", ".srt"],
        "db_path": "./plaud_ingested.db",
        "usb_poll_interval": 5,
        "archive_after_ingest": True,
        "archive_folder": "./PlaudArchive",
    }

    for key, default in defaults.items():
        plaud.setdefault(key, default)

    return {**cfg, "plaud": plaud}


# ---------------------------------------------------------------------------
# Deduplication DB
# ---------------------------------------------------------------------------
class IngestTracker:
    """SQLite-backed tracker to avoid re-ingesting the same file."""

    def __init__(self, db_path: str):
        db = Path(db_path).expanduser()
        db.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingested (
                    file_hash TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    job_id TEXT,
                    method TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _hash_file(path: str) -> str:
        """SHA-256 of the first 64 KB + file size (fast dedup)."""
        h = hashlib.sha256()
        size = os.path.getsize(path)
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()

    def already_ingested(self, path: str) -> bool:
        file_hash = self._hash_file(path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingested WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return row is not None

    def mark_ingested(
        self, path: str, method: str, job_id: Optional[str] = None
    ) -> None:
        file_hash = self._hash_file(path)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingested
                    (file_hash, original_path, filename, ingested_at, job_id, method)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_hash, str(path), Path(path).name, now, job_id, method),
            )


# ---------------------------------------------------------------------------
# Ingestion backends
# ---------------------------------------------------------------------------
def _submit_via_api(file_path: str, api_url: str, dry_run: bool = False) -> Optional[str]:
    """POST the audio file to the ClinicalWhisper FastAPI endpoint."""
    if dry_run:
        log.info("  [DRY-RUN] Would POST %s → %s", Path(file_path).name, api_url)
        return "dry-run-job-id"

    import requests  # lazy import — only needed for API mode

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                api_url,
                files={"file": (Path(file_path).name, f)},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("job_id", "unknown")
        log.info("  ✅ Submitted → job_id=%s  (status=%s)", job_id, data.get("status"))
        return job_id
    except Exception as exc:
        log.error("  ❌ API submission failed: %s", exc)
        return None


def _submit_via_direct(
    file_path: str, input_folder: str, dry_run: bool = False
) -> Optional[str]:
    """Copy the audio file directly into ClinicalWhisper's Input folder."""
    dest = Path(input_folder) / Path(file_path).name
    if dry_run:
        log.info("  [DRY-RUN] Would copy %s → %s", Path(file_path).name, dest)
        return "dry-run-direct"

    try:
        Path(input_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        log.info("  ✅ Copied to Input → %s", dest)
        return f"direct-{Path(file_path).stem}"
    except Exception as exc:
        log.error("  ❌ Direct copy failed: %s", exc)
        return None


def _save_transcript_as_note(
    file_path: str, output_folder: str, dry_run: bool = False
) -> Optional[str]:
    """Convert a pre-transcribed text file into a Markdown note."""
    if dry_run:
        log.info("  [DRY-RUN] Would convert transcript %s → Obsidian note", Path(file_path).name)
        return "dry-run-transcript"

    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        stem = Path(file_path).stem
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        md_content = (
            f"---\n"
            f"title: \"{stem}\"\n"
            f"date: {now}\n"
            f"tags: [plaud, pre-transcribed]\n"
            f"source: plaud-note\n"
            f"---\n\n"
            f"# {stem}\n\n"
            f"{text}\n"
        )

        out = Path(output_folder)
        out.mkdir(parents=True, exist_ok=True)
        note_path = out / f"{stem}.md"
        note_path.write_text(md_content, encoding="utf-8")
        log.info("  ✅ Transcript note → %s", note_path)
        return f"transcript-{stem}"
    except Exception as exc:
        log.error("  ❌ Transcript conversion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------
def _archive_file(file_path: str, archive_folder: str) -> None:
    archive = Path(archive_folder)
    archive.mkdir(parents=True, exist_ok=True)
    dest = archive / Path(file_path).name

    # Handle name collisions
    counter = 1
    while dest.exists():
        dest = archive / f"{Path(file_path).stem}_{counter}{Path(file_path).suffix}"
        counter += 1

    try:
        shutil.move(file_path, str(dest))
        log.info("  📦 Archived → %s", dest)
    except Exception as exc:
        log.warning("  ⚠️  Archive failed: %s", exc)


# ---------------------------------------------------------------------------
# Watchdog handler
# ---------------------------------------------------------------------------
class PlaudFileHandler(FileSystemEventHandler):
    """Handles new audio files appearing in watched directories."""

    def __init__(self, cfg: dict, tracker: IngestTracker, dry_run: bool = False):
        super().__init__()
        self.cfg = cfg
        self.plaud = cfg["plaud"]
        self.tracker = tracker
        self.dry_run = dry_run
        self.audio_exts = set(
            ext.lower() for ext in self.plaud.get("audio_extensions", [])
        )
        self.transcript_exts = set(
            ext.lower() for ext in self.plaud.get("transcript_extensions", [])
        )

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        self._handle(event.dest_path)

    def _handle(self, file_path: str) -> None:
        ext = Path(file_path).suffix.lower()
        is_audio = ext in self.audio_exts
        is_transcript = ext in self.transcript_exts and self.plaud.get(
            "ingest_transcriptions", False
        )

        if not is_audio and not is_transcript:
            return

        # Wait for file to finish writing
        time.sleep(1.5)
        if not os.path.exists(file_path):
            return

        # Minimum size check (skip empty / partial files)
        try:
            if os.path.getsize(file_path) < 1024 and is_audio:
                log.debug("Skipping tiny file: %s", file_path)
                return
        except OSError:
            return

        # Dedup check
        if self.tracker.already_ingested(file_path):
            log.info("⏭️  Already ingested, skipping: %s", Path(file_path).name)
            return

        log.info("🎙️  New Plaud file: %s", file_path)

        job_id = None
        method = "unknown"

        if is_audio:
            mode = self.plaud.get("ingestion_mode", "api")
            if mode == "api":
                job_id = _submit_via_api(
                    file_path, self.plaud["api_url"], self.dry_run
                )
                method = "api"
            else:
                input_folder = resolve_path(self.cfg.get("input_folder", "./Input"))
                job_id = _submit_via_direct(file_path, input_folder, self.dry_run)
                method = "direct"
        elif is_transcript:
            output_folder = resolve_path(self.cfg.get("output_folder", "./Output"))
            job_id = _save_transcript_as_note(file_path, output_folder, self.dry_run)
            method = "transcript"

        if job_id:
            self.tracker.mark_ingested(file_path, method=method, job_id=job_id)

            if self.plaud.get("archive_after_ingest", True) and not self.dry_run:
                archive_folder = resolve_path(
                    self.plaud.get("archive_folder", "./PlaudArchive")
                )
                _archive_file(file_path, archive_folder)


# ---------------------------------------------------------------------------
# USB mount poller
# ---------------------------------------------------------------------------
def _resolve_watch_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns (e.g. /Volumes/PLAUD*) into actual paths."""
    resolved = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        if "*" in expanded or "?" in expanded:
            matches = glob.glob(expanded)
            resolved.extend(m for m in matches if os.path.isdir(m))
        else:
            if os.path.isdir(expanded):
                resolved.append(expanded)
    return list(set(resolved))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(
    config_path: Optional[str] = None,
    watch_override: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    cfg = _load_plaud_config(config_path)
    plaud = cfg["plaud"]

    if not plaud.get("enabled", True):
        log.info("Plaud connector disabled in config. Exiting.")
        return

    db_path = resolve_path(plaud["db_path"])
    tracker = IngestTracker(db_path)

    handler = PlaudFileHandler(cfg, tracker, dry_run=dry_run)

    # Determine watch directories
    if watch_override:
        raw_patterns = [watch_override]
    else:
        raw_patterns = plaud.get("watch_paths", [])

    usb_poll_interval = float(plaud.get("usb_poll_interval", 5))
    observer = Observer()
    active_watches: dict[str, object] = {}

    def _sync_watches() -> None:
        """Add/remove observers as directories appear/disappear (USB plug-in)."""
        current_dirs = set(_resolve_watch_paths(raw_patterns))

        # Remove watches for directories that vanished
        for d in list(active_watches.keys()):
            if d not in current_dirs:
                log.info("📤 Watch removed (unmounted?): %s", d)
                try:
                    observer.unschedule(active_watches[d])
                except Exception:
                    pass
                del active_watches[d]

        # Add watches for new directories
        for d in current_dirs:
            if d not in active_watches:
                log.info("👁️  Watching: %s", d)
                try:
                    watch = observer.schedule(handler, d, recursive=True)
                    active_watches[d] = watch
                except Exception as exc:
                    log.warning("⚠️  Could not watch %s: %s", d, exc)

    # Also ensure static directories exist
    for pattern in raw_patterns:
        expanded = os.path.expanduser(pattern)
        if "*" not in expanded and "?" not in expanded:
            Path(expanded).mkdir(parents=True, exist_ok=True)

    _sync_watches()
    observer.start()

    mode_label = "DRY-RUN" if dry_run else plaud.get("ingestion_mode", "api").upper()
    log.info(
        "🚀 PlaudConnector started (mode=%s, poll=%is)",
        mode_label,
        int(usb_poll_interval),
    )
    log.info("   Watching patterns: %s", raw_patterns)

    try:
        while True:
            time.sleep(usb_poll_interval)
            _sync_watches()
    except KeyboardInterrupt:
        log.info("🛑 Shutting down PlaudConnector...")
        observer.stop()
    observer.join()
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PlaudConnector — auto-ingest Plaud Note recordings into ClinicalWhisper"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to project root)",
    )
    parser.add_argument(
        "--watch",
        type=str,
        default=None,
        help="Override: watch this single directory instead of config paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect files but don't actually submit or archive them",
    )
    args = parser.parse_args()
    run(config_path=args.config, watch_override=args.watch, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
