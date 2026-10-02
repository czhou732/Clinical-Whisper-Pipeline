"""Desktop entry point: starts the local API server and opens the native window.

multiprocessing.freeze_support() MUST run before anything else. In a frozen
macOS app, a spawned child process re-executes this same bundle binary; without
freeze_support the child falls through and boots a *second* server and window.
Any library that touches multiprocessing (torch, tokenizers, openmed) therefore
produced a new ClinicalWhisper instance every run.
"""

import multiprocessing

multiprocessing.freeze_support()

# Must run before uvicorn/gui_server pull in transformers or huggingface_hub:
# both read HF_* from the environment at import time.
import bundled_models  # noqa: E402

bundled_models.configure()

# No network from here on: models load from disk, outgoing connections are refused.
import offline  # noqa: E402

offline.lock()

# `ClinicalWhisper --batch -i <folder> -o <summary.csv> [...]` runs the batch
# processor with the models inside the app: no Python install, no download,
# and the same code as the tested clinicalwhisper-batch command. Checked before
# the window, the server and the crash log (whose file handler would swallow
# the progress lines this mode prints to the terminal).
import sys  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--batch":
    import batch_processor  # noqa: E402

    batch_processor.run(sys.argv[2:], prog="ClinicalWhisper --batch")

# Before anything heavy imports: a crash from here on leaves a log.
import crash_diagnostics  # noqa: E402

crash_diagnostics.configure()

import atexit  # noqa: E402
import fcntl  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import uvicorn  # noqa: E402
import webview  # noqa: E402

import json  # noqa: E402

import gui_server  # noqa: E402
from gui_server import app  # noqa: E402

# A lock file rather than a fixed port: the port is dynamic by design, and two
# windows talking to two servers over the same data directory would race on
# Input/ and Output/.
LOCK_PATH = Path.home() / "Library" / "Application Support" / "ClinicalWhisper" / "app.lock"
# Kept open for the life of the process: closing it would drop the lock.
_LOCK_FILE = None


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def acquire_single_instance() -> bool:
    """Return True if this process may run; False if another window owns it.

    The lock is an advisory lock the kernel holds on an open file, not a
    recorded process id. A recorded id cannot be trusted: after an unclean exit
    the file survives, macOS eventually reuses that number for an unrelated
    process, and every later launch then sees a "live" owner and quits without
    a word — which is what happened here, with the id belonging to
    SystemUIServer. A kernel lock is released when the process ends, however it
    ends, including a crash or a force quit.
    """
    global _LOCK_FILE
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOCK_FILE = open(LOCK_PATH, "a+")
        try:
            fcntl.flock(_LOCK_FILE, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _LOCK_FILE.close()
            _LOCK_FILE = None
            return False  # another running instance holds it
        _LOCK_FILE.seek(0)
        _LOCK_FILE.truncate()
        _LOCK_FILE.write(str(os.getpid()))  # for humans reading the file
        _LOCK_FILE.flush()
        atexit.register(release_single_instance)
        return True
    except OSError:
        # If the lock cannot be managed, prefer running over refusing to start.
        return True


def release_single_instance() -> None:
    global _LOCK_FILE
    try:
        if _LOCK_FILE is not None:
            fcntl.flock(_LOCK_FILE, fcntl.LOCK_UN)
            _LOCK_FILE.close()
            _LOCK_FILE = None
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


FREE_PORT = get_free_port()


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=FREE_PORT, log_level="error")


_AUDIO_TYPES = ("Audio files (*.wav;*.m4a;*.mp3;*.mp4;*.ogg;*.opus;*.flac;*.aac)",)


def _file_entry(path: str) -> dict:
    from progress_report import audio_seconds

    p = Path(path)
    return {"path": str(p), "name": p.name, "size": p.stat().st_size,
            "duration": audio_seconds(p)}


class Bridge:
    """What the app's own page can ask the app to do, outside HTTP.

    pywebview exposes these methods to the page loaded in this window only;
    other web pages cannot reach them. That is what makes it safe to accept
    file paths here, which an HTTP endpoint could not do safely. Private
    attributes (leading underscore) are not exposed to the page.
    """

    def __init__(self):
        self._window = None

    def pick_files(self) -> list:
        """The macOS open dialog; returns real paths, so nothing is uploaded."""
        chosen = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True, file_types=_AUDIO_TYPES)
        return [_file_entry(p) for p in (chosen or [])]

    def start_batch(self, request: dict) -> dict:
        """Process the chosen recordings where they are."""
        try:
            return gui_server.start_batch_from_paths(
                request.get("paths", []),
                participant_id=request.get("participant_id", ""),
                session_label=request.get("session_label", ""),
                criterion_score=request.get("criterion_score", ""),
                transcribe_only=bool(request.get("transcribe_only")),
                num_speakers=request.get("num_speakers"),
            )
        except (OSError, ValueError) as exc:
            return {"status": "error", "message": str(exc)}


def _watch_drops(window) -> None:
    """Hand the page the real paths of files dropped on it.

    macOS only records a dropped file's path when a Python listener is
    attached to the drop target, so attach one each time the page loads.
    """
    def on_drop(event):
        files = (event.get("dataTransfer") or {}).get("files") or []
        entries = [_file_entry(f["pywebviewFullPath"]) for f in files
                   if f.get("pywebviewFullPath")]
        if entries:
            window.evaluate_js(f"window.cwAddPaths && window.cwAddPaths({json.dumps(entries)})")

    def on_loaded():
        zone = window.dom.get_element("#drop-zone")
        if zone is not None:
            zone.events.drop += on_drop

    window.events.loaded += on_loaded


if __name__ == "__main__":
    if not acquire_single_instance():
        # Another window is already open; bring it forward instead of stacking
        # a second server on the same data directory. Touch nothing else: the
        # session marker belongs to the copy that is running.
        if sys.platform == "darwin":
            os.system("open -a ClinicalWhisper 2>/dev/null")
        sys.exit(0)
    crash_diagnostics.start_session()

    # An app launched from Finder inherits a minimal PATH that excludes Homebrew,
    # so ffmpeg would be invisible to the pipeline's subprocess calls.
    os.environ["PATH"] += os.pathsep + "/usr/local/bin" + os.pathsep + "/opt/homebrew/bin"

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    bridge = Bridge()
    window = webview.create_window(
        "ClinicalWhisper",
        # ?app=1 tells the page it is in the app window, before the bridge
        # is injected, so it never falls back to uploading files.
        f"http://127.0.0.1:{FREE_PORT}/?app=1",
        js_api=bridge,
        width=1000,
        height=820,
        resizable=True,
        text_select=True,
        background_color="#FAFAFA",
    )
    bridge._window = window
    _watch_drops(window)
    webview.start()
    crash_diagnostics.shutdown()
    release_single_instance()
    # Skip native teardown on quit: it can abort (see batch_processor.exit_now),
    # which macOS reports as "quit unexpectedly".
    from batch_processor import exit_now
    exit_now(0)
