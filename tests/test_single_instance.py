"""The app's single-instance lock.

A stale lock must never block a launch: the file survives an unclean exit, and
macOS reuses process ids, so a recorded id eventually points at an unrelated
process (in one case SystemUIServer) and the app quit silently on every launch.
"""

import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    import launcher as mod

    monkeypatch.setattr(mod, "LOCK_PATH", tmp_path / "app.lock")
    monkeypatch.setattr(mod, "_LOCK_FILE", None)
    yield mod
    mod.release_single_instance()


def test_launches_with_no_lock_file(launcher):
    assert launcher.acquire_single_instance()


def test_stale_lock_naming_a_live_unrelated_process_does_not_block(launcher):
    # PID 1 (launchd) is always alive and is never ClinicalWhisper.
    launcher.LOCK_PATH.write_text("1")
    assert launcher.acquire_single_instance()


def test_second_instance_is_refused_while_the_first_holds_the_lock(launcher):
    assert launcher.acquire_single_instance()
    holder = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import fcntl
            f = open({str(launcher.LOCK_PATH)!r}, "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("acquired")
            except OSError:
                print("refused")
        """)],
        capture_output=True, text=True, timeout=30,
    )
    assert holder.stdout.strip() == "refused"


def test_lock_is_released_when_the_process_ends(tmp_path):
    """Even on a hard kill: the kernel drops the lock when the process dies."""
    lock = tmp_path / "app.lock"
    script = textwrap.dedent(f"""
        import fcntl, os, time
        f = open({str(lock)!r}, "a+")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("holding", flush=True)
        time.sleep(30)
    """)
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "holding"
    child.kill()
    child.wait(timeout=10)

    import fcntl
    with open(lock, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # no exception: lock is free
        fcntl.flock(f, fcntl.LOCK_UN)


def test_app_runs_as_a_batch_command():
    """`ClinicalWhisper --batch` reaches the batch processor, not the window."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "launcher.py"), "--batch", "--help"],
                       capture_output=True, text=True, timeout=120, cwd=root)
    assert r.returncode == 0
    assert "ClinicalWhisper --batch" in r.stdout and "--transcribe-only" in r.stdout
