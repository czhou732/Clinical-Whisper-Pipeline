"""Make crashes leave evidence.

An app launched from Finder has no terminal, so without this every log line
and traceback went nowhere. Worse, a native crash (a Metal/MLX abort, or macOS
killing the process for memory) closes the window with no Python traceback at
all — which is exactly what a collaborator reported: "the platform just
crashed and closed entirely."

This module, configured first thing at launch:

* writes a rotating log to ``~/Library/Logs/ClinicalWhisper/clinicalwhisper.log``
  with a header describing the machine (macOS version, chip, RAM);
* routes uncaught exceptions from any thread into that log;
* enables :mod:`faulthandler`, which dumps the Python stack of every thread to
  ``crash.log`` on SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL;
* keeps a session marker recording the stage in progress, so the next launch
  can report that the previous run died, and where.

A SIGKILL (the out-of-memory killer) cannot be intercepted by anything, but
the session marker still records what the app was doing when it happened.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

LOG_DIR = Path.home() / "Library" / "Logs" / "ClinicalWhisper"
LOG_PATH = LOG_DIR / "clinicalwhisper.log"
CRASH_PATH = LOG_DIR / "crash.log"
SESSION_PATH = LOG_DIR / "session.json"

log = logging.getLogger("ClinicalWhisper")

_crash_file = None
_previous_crash: Optional[dict] = None
_lock = threading.Lock()


def machine_summary() -> dict:
    """What a bug report needs to know about the machine."""
    info = {
        "macos": platform.mac_ver()[0] or platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError):
        pass
    try:
        info["chip"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_session(**fields) -> None:
    with _lock:
        try:
            data = json.loads(SESSION_PATH.read_text()) if SESSION_PATH.exists() else {}
            data.update(fields)
            SESSION_PATH.write_text(json.dumps(data))
        except (OSError, ValueError):
            pass


def configure() -> None:
    """Set up logging, crash capture and the session marker. Call once."""
    global _crash_file, _previous_crash
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    machine = machine_summary()
    log.info("=== ClinicalWhisper session start (pid %d) %s", os.getpid(), json.dumps(machine))

    _crash_file = open(CRASH_PATH, "a", buffering=1, encoding="utf-8")
    _crash_file.write(f"\n=== session {time.ctime()} pid {os.getpid()} {json.dumps(machine)}\n")
    faulthandler.enable(file=_crash_file, all_threads=True)

    def _excepthook(exc_type, exc, tb):
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_excepthook(args):
        log.critical(
            "Uncaught exception in thread %s", args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


def start_session() -> None:
    """Claim the session marker, reporting the previous session if it died mid-work.

    Call only once this process knows it is the running instance: a second
    launch that exits straight away must not overwrite the marker of the copy
    that is actually working.

    A previous session is reported only if it was processing when it ended
    (``working``). Quitting or killing an idle app is not a crash, and stage
    names can outlive the work they describe, so they are not evidence alone.
    """
    global _previous_crash
    try:
        prev = json.loads(SESSION_PATH.read_text()) if SESSION_PATH.exists() else None
    except (OSError, ValueError):
        prev = None
    if (prev and prev.get("working") and prev.get("pid") != os.getpid()
            and not _pid_alive(int(prev.get("pid", 0)))):
        _previous_crash = prev
        log.error("Previous session stopped while processing: %s", json.dumps(prev))
    SESSION_PATH.write_text(json.dumps({"pid": os.getpid(), "started": time.time(),
                                        "stage": None, "working": False}))


def begin_work(detail: str = "") -> None:
    """A batch has started: an exit before :func:`end_work` is worth reporting."""
    _write_session(working=True, stage=None, detail=detail, at=time.time())


def end_work() -> None:
    """The batch is over (finished, failed or cancelled)."""
    _write_session(working=False, stage=None, detail="", at=time.time())


def mark_stage(stage: Optional[str], detail: str = "") -> None:
    """Record what the app is doing, so a hard crash can be located later."""
    _write_session(stage=stage, detail=detail, at=time.time())
    if stage:
        log.info("Stage: %s %s", stage, detail)


def previous_crash() -> Optional[dict]:
    """The unfinished stage of the previous session, if it crashed."""
    if _previous_crash is None:
        return None
    return {**_previous_crash, "log_path": str(LOG_PATH), "crash_log_path": str(CRASH_PATH)}


def dismiss_previous_crash() -> None:
    global _previous_crash
    _previous_crash = None


def shutdown() -> None:
    """Clean exit: clear the marker so the next launch does not report a crash."""
    try:
        SESSION_PATH.unlink()
    except OSError:
        pass
    log.info("=== ClinicalWhisper session end (pid %d)", os.getpid())
