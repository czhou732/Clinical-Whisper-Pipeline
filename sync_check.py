"""Is a folder synced to a cloud service?

ClinicalWhisper sends nothing over the network, but macOS and sync clients
can: with iCloud's "Desktop & Documents" option on, everything written under
~/Documents is uploaded to Apple, and OneDrive, Box, Dropbox and Google Drive
do the same for their folders. Transcripts and audio copies written there
leave the machine even though the app never connects to anything.

Detection is best effort, by two independent signals:
* macOS File Provider tags each synced folder with the extended attribute
  ``com.apple.file-provider-domain-id`` (iCloud Drive, including a synced
  Documents or Desktop, and sync clients built on File Provider);
* well-known sync locations under the home folder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

_FILE_PROVIDER_ATTR = "com.apple.file-provider-domain-id"

# Home-relative folders that sync clients use (lower-cased first component).
_KNOWN = {
    "library/mobile documents": "iCloud Drive",
    "library/cloudstorage": "a cloud storage provider (OneDrive, Box, Google Drive or Dropbox)",
    "dropbox": "Dropbox",
    "box": "Box",
    "box sync": "Box",
    "google drive": "Google Drive",
}


def _has_file_provider_tag(folder: Path) -> bool:
    try:
        r = subprocess.run(["xattr", "-p", _FILE_PROVIDER_ATTR, str(folder)],
                           capture_output=True, timeout=3)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def synced_by(path: Path, home: Optional[Path] = None) -> Optional[str]:
    """The service that syncs ``path``, or None if it looks local."""
    home = (home or Path.home()).resolve()
    path = Path(path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    try:
        rel = path.relative_to(home)
    except ValueError:
        rel = None
    if rel is not None and rel.parts:
        lowered = "/".join(rel.parts).lower()
        for prefix, service in _KNOWN.items():
            if lowered == prefix or lowered.startswith(prefix + "/"):
                return service
        if rel.parts[0].lower().startswith("onedrive"):
            return "OneDrive"
    # Walk up to the home folder looking for a File Provider tag.
    folder = path if path.is_dir() else path.parent
    for candidate in [folder, *folder.parents]:
        if candidate == home or candidate == candidate.parent:
            break
        if candidate.exists() and _has_file_provider_tag(candidate):
            return "iCloud Drive or another sync service"
    return None


def warning_for(path: Path) -> Optional[str]:
    service = synced_by(path)
    if not service:
        return None
    return (f"{path} is synced by {service}. Files written there are uploaded off this "
            "computer, including transcripts. Choose a folder that is not synced, such as "
            f"{Path.home() / 'ClinicalWhisper'}.")
