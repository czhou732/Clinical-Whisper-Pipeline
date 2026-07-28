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

import atexit  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import uvicorn  # noqa: E402
import webview  # noqa: E402

from gui_server import app  # noqa: E402

# A lock file rather than a fixed port: the port is dynamic by design, and two
# windows talking to two servers over the same data directory would race on
# Input/ and Output/.
LOCK_PATH = Path.home() / "Library" / "Application Support" / "ClinicalWhisper" / "app.lock"


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def acquire_single_instance() -> bool:
    """Return True if this process may run; False if another instance owns it."""
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOCK_PATH.exists():
            try:
                existing = int(LOCK_PATH.read_text().strip())
            except (ValueError, OSError):
                existing = -1
            if existing > 0 and existing != os.getpid() and _pid_alive(existing):
                return False
        LOCK_PATH.write_text(str(os.getpid()))
        atexit.register(release_single_instance)
        return True
    except OSError:
        # If the lock cannot be managed, prefer running over refusing to start.
        return True


def release_single_instance() -> None:
    try:
        if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
            LOCK_PATH.unlink()
    except OSError:
        pass


FREE_PORT = get_free_port()


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=FREE_PORT, log_level="error")


if __name__ == "__main__":
    if not acquire_single_instance():
        # Another window is already open; bring it forward instead of stacking
        # a second server on the same data directory.
        if sys.platform == "darwin":
            os.system("open -a ClinicalWhisper 2>/dev/null")
        sys.exit(0)

    # An app launched from Finder inherits a minimal PATH that excludes Homebrew,
    # so ffmpeg would be invisible to the pipeline's subprocess calls.
    os.environ["PATH"] += os.pathsep + "/usr/local/bin" + os.pathsep + "/opt/homebrew/bin"

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    webview.create_window(
        "ClinicalWhisper",
        f"http://127.0.0.1:{FREE_PORT}",
        width=1000,
        height=820,
        resizable=True,
        text_select=True,
        background_color="#FAFAFA",
    )
    webview.start()
    release_single_instance()
