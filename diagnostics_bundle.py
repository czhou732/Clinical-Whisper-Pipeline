"""One zip a collaborator can email when something goes wrong.

The logs never contain audio or transcript text, but they do name files and
folders, and a file name can carry a participant ID or a person's name. So
every path, every file name with an audio or result extension, and the Mac's
user name are replaced before anything is zipped. What is left is timings,
stages, warnings, stack traces and a description of the machine.
"""

from __future__ import annotations

import getpass
import json
import re
import time
import zipfile
from pathlib import Path

import crash_diagnostics

_EXTENSIONS = r"(?:wav|mp3|m4a|mp4|mov|flac|ogg|opus|aac|webm|json|csv|txt|docx|pdf)"
# A path runs to the end of the line or the next quote, so folder names with
# spaces ("journal club audio/…") are removed whole.
_PATH = re.compile(r"(?:~|/(?:Users|private|var|Volumes|tmp|Library|Applications))/[^'\"\n]*")
# A file name may contain spaces, so the words before it on the line go too
# (after the last colon): losing a word of context is safer than leaking a name.
_FILE = re.compile(r"[^\s'\"/:]+(?: [^\s'\"/:]+)*\." + _EXTENSIONS + r"\b", re.IGNORECASE)

README = """ClinicalWhisper diagnostics

Contains: app logs, crash stack traces, and a summary of this Mac (macOS
version, chip, memory, app version). File names, folder paths and the
account name have been replaced with <path>, <file> and <user>.

Does not contain: audio, transcripts, clinical scores, or results files.
"""


def redact(text: str, user: str | None = None) -> str:
    """Remove paths, file names and the account name from log text."""
    user = user or getpass.getuser()
    text = _PATH.sub("<path>", text)
    text = _FILE.sub("<file>", text)
    if len(user) > 2:
        text = re.sub(re.escape(user), "<user>", text, flags=re.IGNORECASE)
    return text


def save(dest_dir: Path, extra: dict | None = None) -> Path:
    """Write a redacted diagnostics zip into ``dest_dir`` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / time.strftime("ClinicalWhisper-diagnostics-%Y%m%d-%H%M%S.zip")
    summary = {**crash_diagnostics.machine_summary(), **(extra or {})}
    # Only this app's files: other tools may log into the same folder.
    own = {crash_diagnostics.LOG_PATH.name, crash_diagnostics.CRASH_PATH.name,
           crash_diagnostics.SESSION_PATH.name}
    logs = sorted(p for p in crash_diagnostics.LOG_DIR.glob("*")
                  if p.is_file() and (p.name in own
                                      or p.name.startswith(crash_diagnostics.LOG_PATH.name + ".")))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", README)
        zf.writestr("machine.json", redact(json.dumps(summary, indent=2, default=str)))
        for path in logs:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            zf.writestr(f"logs/{path.name}", redact(text))
    return out
