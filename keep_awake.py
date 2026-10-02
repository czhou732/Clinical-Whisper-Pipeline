"""Keep the Mac from sleeping while recordings are being processed.

A multi-hour batch on a laptop otherwise stops whenever the machine idles into
sleep, and a run left overnight does nothing for most of the night. This uses
macOS's own ``caffeinate``: -i blocks idle sleep, -s blocks system sleep on
mains power, and -w ties it to this process, so it ends even if the app
crashes. It cannot keep a laptop awake with the lid closed on battery; the
quickstart says to keep it plugged in with the lid open.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys

log = logging.getLogger("ClinicalWhisper")


@contextlib.contextmanager
def keep_awake():
    proc = None
    if sys.platform == "darwin":
        try:
            proc = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:  # never block processing over this
            log.warning("Could not keep the Mac awake: %s", exc)
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
