"""A macOS notification when a batch finishes, so a long run need not be watched."""

from __future__ import annotations

import subprocess
import sys


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(title: str, message: str) -> None:
    """Show a notification. Keep file names and transcript text out of it:
    notifications can appear on a locked screen."""
    if sys.platform != "darwin":
        return
    script = (f"display notification {_applescript_string(message)} "
              f"with title {_applescript_string(title)}")
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
