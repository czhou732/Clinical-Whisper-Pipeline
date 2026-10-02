"""FastAPI backend for the ClinicalWhisper desktop GUI.

Serves the static frontend from the app bundle and runs the inference pipeline
over one or more uploaded files, streaming stage-level progress back to the UI.

A batch is a single job queue processed by one worker thread: the MOSS and LLM
weights load once and stay resident for every file in the batch, which is where
most of the per-file time would otherwise go.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import addons
import bundled_models
import crash_diagnostics

# Harmless when running from source; redirects to the in-bundle weights when
# frozen. Runs before the pipeline imports transformers.
bundled_models.configure()

import offline  # noqa: E402

offline.lock()  # idempotent; the launcher has usually done it already

from cw_config import AUDIO_EXTENSIONS, DATA_ROOT, LEGACY_DATA_ROOT, load_config
from mask_legend import legend as mask_legend
from transcript_formatter import speaker_samples

app = FastAPI(title="ClinicalWhisper GUI Server")

# The server listens on this machine only, but any web page open in the
# user's browser can still send requests to 127.0.0.1. So:
# * no CORS headers: other sites cannot read responses;
# * requests that change state must come from this app's own page;
# * the Host header must name this machine, which defeats DNS rebinding
#   (a site pointing its own hostname at 127.0.0.1 to look same-origin).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def _same_origin_only(request, call_next):
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
        if origin != f"http://{request.headers.get('host', '')}":
            return JSONResponse(status_code=403, content={
                "status": "error", "message": "Request from another site refused."})
    return await call_next(request)

# Code and static assets live in the (read-only) bundle; user data does not.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
WWW_DIR = BUNDLE_DIR / "www"

INPUT_DIR = DATA_ROOT / "Input"
OUTPUT_DIR = DATA_ROOT / "Output"
for _d in (INPUT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("ClinicalWhisper.gui")


def _config_path() -> str:
    """Prefer a user-editable config, fall back to the bundled example."""
    user_cfg = DATA_ROOT / "config.yaml"
    if user_cfg.exists():
        return str(user_cfg)
    return str(BUNDLE_DIR / "config.example.yaml")


# ── Batch state ─────────────────────────────────────────────────────────────

_lock = threading.Lock()
_batches: dict[str, dict] = {}


def _new_batch(files: list[Path]) -> str:
    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    with _lock:
        _batches[batch_id] = {
            "status": "Queued",
            "log": ["Queued."],
            "total": len(files),
            "done": 0,
            "current": None,
            "cancelled": False,
            "stage": None,
            "stage_fraction": None,
            "stage_detail": "",
            "files": [
                {"filename": f.name, "state": "pending", "result": None,
                 "warnings": [], "error": None, "analysis_path": None,
                 "transcript": "", "structured_transcript": "", "analysis": None}
                for f in files
            ],
        }
    return batch_id


class _BatchLogHandler(logging.Handler):
    """Mirrors pipeline log records into the batch's log buffer for the GUI."""

    def __init__(self, batch_id: str):
        super().__init__(level=logging.INFO)
        self.batch_id = batch_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = record.getMessage()
        except Exception:
            return
        with _lock:
            batch = _batches.get(self.batch_id)
            if batch is None:
                return
            batch["log"].append(line)
            del batch["log"][:-300]
            batch["status"] = line


def process_batch_task(batch_id: str, paths: list[Path]) -> None:
    """Run every file in the batch through one warm pipeline instance."""
    handler = _BatchLogHandler(batch_id)
    cw_log = logging.getLogger("ClinicalWhisper")
    cw_log.addHandler(handler)
    cw_log.setLevel(logging.INFO)
    crash_diagnostics.begin_work(f"{len(paths)} file(s)")
    from keep_awake import keep_awake

    awake = keep_awake()
    awake.__enter__()
    try:
        # Imported here, inside the try: a missing dependency in a frozen build
        # would otherwise raise before any state is set, leaving the UI stuck.
        from batch_processor import _extract_row
        from inference_pipeline import InferencePipeline

        cfg = load_config(_config_path())
        with _lock:
            if _batches.get(batch_id, {}).get("num_speakers"):
                # Extra speaker labels are folded into this many by voice.
                cfg.setdefault("moss", {})["num_speakers"] = _batches[batch_id]["num_speakers"]
            if _batches.get(batch_id, {}).get("in_place"):
                # The user's own recording, read where it is: never move or delete it.
                cfg["audio_retention"] = "keep"
            if _batches.get(batch_id, {}).get("transcribe_only"):
                cfg.setdefault("llm_scoring", {})["enabled"] = False
                cfg["llm_scoring"]["skipped_by_request"] = True
                _batches[batch_id]["log"].append(
                    "Transcribe only: clinical scoring is off for this batch.")

        def _progress(stage, fraction, detail):
            with _lock:
                b = _batches.get(batch_id)
                if b is not None:
                    b["stage"] = stage
                    b["stage_fraction"] = fraction
                    b["stage_detail"] = detail

        def _cancelled():
            with _lock:
                b = _batches.get(batch_id)
                return bool(b and b["cancelled"])

        pipeline = InferencePipeline(cfg, progress_cb=_progress, should_cancel=_cancelled)

        with _lock:
            edits_per_file = list(_batches[batch_id].get("edits") or [])
            meta = (
                _batches[batch_id].get("participant_id", ""),
                _batches[batch_id].get("session_label", ""),
                _batches[batch_id].get("criterion_score", ""),
            )

        # Two phases so the transcription and scoring models are never
        # resident together: MOSS is ~1.7 GB and Llama-3 ~4.9 GB, and holding
        # both put peak model memory at ~6.6 GB for no benefit.
        states: list[tuple[int, dict]] = []
        rows: list[dict] = []

        # ── Phase 1: transcribe every file, together ──
        # Windows from all files share decoding batches, so a folder of short
        # recordings runs at batch speed rather than one file at a time.
        jobs = []
        for idx, audio_path in enumerate(paths):
            jobs.append({
                "job_id": f"gui_{uuid.uuid4().hex[:12]}",
                "file_path": str(audio_path),
                "original_filename": audio_path.name,
                # Blank ID with several files: each file is its own participant,
                # named by its filename, rather than all sharing one blank ID.
                "participant_id": meta[0] or (audio_path.stem if len(paths) > 1 else ""),
                "session_label": meta[1],
                "criterion_score": meta[2],
                "audio_edits": edits_per_file[idx] if idx < len(edits_per_file) else None,
            })
        with _lock:
            b = _batches[batch_id]
            b["current"] = f"{len(paths)} file(s)"
            for idx, audio_path in enumerate(paths):
                b["files"][idx]["state"] = "running"
            b["log"].append(f"Transcribing {len(paths)} file(s) together...")

        import speed_model
        from progress_report import audio_seconds

        durations = [audio_seconds(p) for p in paths]
        t_phase = time.monotonic()
        outcomes = [] if _cancelled() else pipeline.transcribe_jobs(jobs)
        speed_model.record("transcribe", sum(
            d or 0 for d, o in zip(durations, outcomes) if not isinstance(o, Exception)),
            time.monotonic() - t_phase)
        for idx, outcome in enumerate(outcomes):
            audio_path = paths[idx]
            if not isinstance(outcome, Exception):
                states.append((idx, outcome))
                continue
            if _cancelled():
                with _lock:
                    _batches[batch_id]["files"][idx]["state"] = "cancelled"
                continue
            log.warning("Batch %s: %s failed: %s", batch_id, audio_path.name, outcome)
            with _lock:
                f = _batches[batch_id]["files"][idx]
                f["state"] = "error"
                f["error"] = str(outcome) or outcome.__class__.__name__
                _batches[batch_id]["done"] += 1
                _batches[batch_id]["log"].append(f"ERROR ({audio_path.name}): {outcome}")
        if _cancelled():
            with _lock:
                _batches[batch_id]["log"].append("Stopped during transcription.")
                for f in _batches[batch_id]["files"]:
                    if f["state"] == "running":
                        f["state"] = "cancelled"

        # Hand back the transcription weights before the scoring model loads.
        pipeline.release_transcriber()

        # ── Phase 2: score every transcript ──
        for idx, state in states:
            if _cancelled():
                with _lock:
                    _batches[batch_id]["files"][idx]["state"] = "cancelled"
                break

            audio_path = paths[idx]
            with _lock:
                _batches[batch_id]["current"] = audio_path.name

            try:
                t_score = time.monotonic()
                analysis_path = pipeline.score_job(state)
                if cfg.get("llm_scoring", {}).get("enabled", True):
                    speed_model.record("score", durations[idx], time.monotonic() - t_score)
                row = _extract_row(analysis_path, audio_path.name)

                with open(analysis_path, "r", encoding="utf-8") as fh:
                    analysis = json.load(fh)

                rows.append(row)
                with _lock:
                    f = _batches[batch_id]["files"][idx]
                    f["state"] = "done"
                    f["result"] = row
                    f["warnings"] = analysis.get("warnings", [])
                    f["analysis_path"] = analysis_path
                    f["transcript"] = analysis.get("transcript", "")
                    f["structured_transcript"] = analysis.get("structured_transcript", "")
                    f["analysis"] = analysis
                    f["quality"] = analysis.get("quality") or {}
                    f["score_reliability"] = analysis.get("score_reliability") or {}
                    # Kept so roles can be corrected and the file re-scored
                    # without paying for transcription again.
                    f["segments"] = analysis.get("segments", [])
                    f["speaker_roles"] = analysis.get("speaker_roles", {})
                    f["speaker_names"] = analysis.get("speaker_names", {})
                    f["speaker_samples"] = speaker_samples(f["segments"])
                    _batches[batch_id]["done"] += 1
            except Exception as exc:
                log.warning("Batch %s: scoring %s failed: %s",
                            batch_id, audio_path.name, exc)
                with _lock:
                    f = _batches[batch_id]["files"][idx]
                    f["state"] = "error"
                    f["error"] = str(exc) or exc.__class__.__name__
                    _batches[batch_id]["done"] += 1
                    _batches[batch_id]["log"].append(f"ERROR ({audio_path.name}): {exc}")


        # Idle apps should not sit on gigabytes of weights. Set
        # keep_models_loaded: true to trade memory for a faster next batch.
        if not cfg.get("keep_models_loaded", False):
            pipeline.release_all()
            with _lock:
                _batches[batch_id]["log"].append("Models released from memory.")

        # One combined CSV for the whole batch, alongside the per-file JSON.
        csv_path = None
        if rows:
            stem = paths[0].stem if len(paths) == 1 else f"batch_{len(rows)}_files"
            csv_path = OUTPUT_DIR / f"{stem}_summary.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

        with _lock:
            b = _batches[batch_id]
            b["csv_path"] = str(csv_path) if csv_path else None
            b["current"] = None
            failed = sum(1 for f in b["files"] if f["state"] == "error")
            for f in b["files"]:
                if f["state"] in ("pending", "running"):
                    f["state"] = "cancelled" if b["cancelled"] else "error"
            if b["cancelled"]:
                b["status"] = "CANCELLED"
                b["log"].append(f"Stopped. {len(rows)} file(s) finished before cancelling.")
            else:
                b["status"] = "COMPLETED" if failed == 0 else f"COMPLETED ({failed} failed)"
                b["log"].append(f"Batch finished: {len(rows)} succeeded, {failed} failed.")
    except Exception as e:
        with _lock:
            b = _batches.get(batch_id)
            if b is not None:
                b["status"] = f"ERROR: {e}"
                b["log"].append(f"ERROR: {e}")
    finally:
        # However the batch ended, the app is idle again: quitting now is not a crash.
        crash_diagnostics.end_work()
        with _lock:
            b = _batches.get(batch_id) or {"files": []}
            done = sum(1 for f in b["files"] if f.get("state") == "done")
            failed = sum(1 for f in b["files"] if f.get("state") == "error")
        from notify import notify
        notify("ClinicalWhisper", f"Finished: {done} file(s) done, {failed} failed.")
        awake.__exit__(None, None, None)
        cw_log.removeHandler(handler)




def _speaker_count(value) -> int | None:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None  # "Not sure, or a group": leave the count to the model
    return n if 1 <= n <= 10 else None


def _checked_edits(edits, count: int) -> list:
    """Per-file edits from the window, validated; one entry (or None) per file."""
    import audio_edits

    edits = list(edits or [])
    if edits and len(edits) != count:
        raise ValueError("Edits don't match the list of files.")
    out = []
    for e in edits or [None] * count:
        parsed = audio_edits.Edits.from_dict(e) if e else None
        out.append(parsed.as_dict() if parsed else None)
    return out


# Recordings the window may play while the user marks what to leave out:
# only files the user picked through the app, each behind a random token.
_previews: dict[str, Path] = {}


def register_preview(path: Path) -> str:
    token = uuid.uuid4().hex
    with _lock:
        _previews[token] = Path(path)
    return token


@app.get("/api/preview/{token}")
async def preview_audio(token: str):
    from fastapi.responses import FileResponse

    with _lock:
        path = _previews.get(token)
    if path is None or not path.is_file():
        return JSONResponse(status_code=404, content={"status": "error",
                                                      "message": "Unknown recording."})
    return FileResponse(path)


_peaks_cache: dict[str, list[float]] = {}


def _peaks(path: Path, n: int) -> list[float]:
    """Loudness of ``n`` equal slices of a recording, 0-1, for drawing it.

    Decoded at 4 kHz: enough to see speech and silence, and several times
    faster than the 16 kHz decode used for processing.
    """
    import av
    import numpy as np

    from progress_report import audio_seconds

    duration = audio_seconds(path) or 0.0
    rate = 4000
    total = int(duration * rate) or 1
    sums = np.zeros(n)
    counts = np.zeros(n)
    pos = 0
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                x = out.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                idx = np.minimum(((pos + np.arange(x.size)) * n) // total, n - 1)
                np.add.at(sums, idx, x * x)
                np.add.at(counts, idx, 1)
                pos += x.size
    rms = np.sqrt(sums / np.maximum(counts, 1))
    top = float(np.percentile(rms, 98)) or 1.0
    return [round(float(v), 3) for v in np.minimum(rms / top, 1.0)]


@app.get("/api/peaks/{token}")
async def preview_peaks(token: str, n: int = 160):
    with _lock:
        path = _previews.get(token)
    if path is None or not path.is_file():
        return JSONResponse(status_code=404, content={"status": "error",
                                                      "message": "Unknown recording."})
    n = max(20, min(int(n), 600))
    key = f"{token}:{n}"
    if key not in _peaks_cache:
        try:
            _peaks_cache[key] = await run_in_threadpool(_peaks, path, n)
        except Exception as exc:  # noqa: BLE001 - a picture is optional
            log.info("No waveform for a picked file: %s", exc)
            return JSONResponse(status_code=422, content={"status": "error",
                                                          "message": "Can't read this file."})
    return {"peaks": _peaks_cache[key]}


def _launch_batch(paths: list[Path], participant_id: str, session_label: str,
                  criterion_score: str, transcribe_only: bool, in_place: bool,
                  num_speakers=None, edits=None) -> str:
    edits = _checked_edits(edits, len(paths))
    batch_id = _new_batch(paths)
    with _lock:
        b = _batches[batch_id]
        b["edits"] = edits
        b["num_speakers"] = _speaker_count(num_speakers)
        b["participant_id"] = participant_id.strip()
        b["session_label"] = session_label.strip()
        b["criterion_score"] = criterion_score.strip()
        b["transcribe_only"] = bool(transcribe_only)
        b["in_place"] = in_place
    # An explicit daemon thread rather than FastAPI BackgroundTasks: a batch
    # runs for minutes, which would pin an anyio threadpool slot for its
    # whole duration, and in the frozen app the task was observed never
    # being dispatched at all, leaving the UI stuck on "Queued".
    threading.Thread(
        target=process_batch_task, args=(batch_id, paths),
        name=f"cw-{batch_id}", daemon=True,
    ).start()
    return batch_id


def start_batch_from_paths(paths: list[str], participant_id: str = "",
                           session_label: str = "", criterion_score: str = "",
                           transcribe_only: bool = False, num_speakers=None,
                           edits=None) -> dict:
    """Process recordings where they are, without copying them.

    Called by the app window through pywebview's private bridge, never over
    HTTP: an HTTP endpoint taking file paths would let any web page make the
    app read audio from anywhere on the disk. The recordings are left exactly
    where they are; the usual archive-after-processing step is skipped.
    """
    files = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Not a file: {raw}")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Not a supported audio file: {path.name}")
        files.append(path)
    if not files:
        raise ValueError("No files selected.")
    batch_id = _launch_batch(files, participant_id, session_label, criterion_score,
                             transcribe_only, in_place=True, num_speakers=num_speakers,
                             edits=edits)
    # Tokens first: register_preview takes the same (non-reentrant) lock.
    tokens = [register_preview(path) for path in files]
    with _lock:
        for f, token in zip(_batches[batch_id]["files"], tokens):
            f["preview"] = token
    return {"status": "success", "batch_id": batch_id, "count": len(files),
            "filenames": [p.name for p in files]}


# ── API ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    participant_id: str = Form(""),
    session_label: str = Form(""),
    criterion_score: str = Form(""),
    transcribe_only: str = Form(""),
    num_speakers: str = Form(""),
    edits: str = Form(""),
):
    """Accept one or many audio files and start a single batch job.

    ``transcribe_only`` skips clinical scoring. On a multi-hour interview the
    scoring stage costs far more time than the transcript, and its scores are
    the least reliable part of the output, so a transcription run should not
    have to wait for it.
    """
    try:
        saved: list[Path] = []
        # One folder per upload, and a suffix for repeated names within it: a
        # browser sends only file names, so P01/session.m4a and P02/session.m4a
        # arrive as two "session.m4a" and would otherwise overwrite each other.
        upload_dir = INPUT_DIR / uuid.uuid4().hex[:12]
        upload_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            safe_name = Path(f.filename or "audio").name
            dest = upload_dir / safe_name
            n = 2
            while dest.exists():
                dest = upload_dir / f"{Path(safe_name).stem} ({n}){Path(safe_name).suffix}"
                n += 1
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved.append(dest)

        if not saved:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No files received."},
            )

        batch_id = _launch_batch(
            saved, participant_id, session_label, criterion_score,
            transcribe_only.strip().lower() in ("1", "true", "on", "yes"),
            in_place=False,
            num_speakers=num_speakers,
            edits=json.loads(edits) if edits.strip() else None,
        )
        return {
            "status": "success",
            "batch_id": batch_id,
            "count": len(saved),
            "filenames": [p.name for p in saved],
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


def _speed() -> dict:
    import speed_model
    data = speed_model.load()
    return {"transcribe": data["transcribe"], "score": data["score"],
            "rough": data.get("runs", 0) < 3}


def _version() -> str:
    from version import __version__
    return __version__


def _sync_warning(path: Path):
    from sync_check import warning_for
    return warning_for(path)


def _filevault_on():
    from clinical_safeguards import filevault_on

    return filevault_on()


@app.get("/api/diagnostics")
async def diagnostics():
    """Report whether the previous session crashed, and where the logs are."""
    return {
        "previous_crash": crash_diagnostics.previous_crash(),
        "log_path": str(crash_diagnostics.LOG_PATH),
        "machine": crash_diagnostics.machine_summary(),
        # False means outputs are written to an unencrypted disk.
        "filevault": _filevault_on(),
        "version": _version(),
        "data_root": str(DATA_ROOT),
        # Set when results would be uploaded by a sync service.
        "data_root_synced": _sync_warning(DATA_ROOT),
        # Results from before 5.2 may still sit in a synced ~/Documents folder.
        "legacy_synced": (_sync_warning(LEGACY_DATA_ROOT)
                          if LEGACY_DATA_ROOT.exists() and LEGACY_DATA_ROOT != DATA_ROOT else None),
        "ram_gb": crash_diagnostics.machine_summary().get("ram_gb"),
        "speed": _speed(),
        # False on the base app until the scoring add-on is installed.
        "scoring_available": addons.scoring_available(),
        "languages_available": addons.languages_available(),
    }


@app.post("/api/diagnostics/dismiss")
async def dismiss_diagnostics():
    crash_diagnostics.dismiss_previous_crash()
    return {"ok": True}


@app.post("/api/results/reveal")
def reveal_results():
    """Show the results folder in Finder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["open", str(OUTPUT_DIR)], check=False)
    return {"ok": True, "path": str(OUTPUT_DIR)}


@app.post("/api/diagnostics/save")
def save_diagnostics():
    """Zip redacted logs and a machine summary into the data folder, then show it."""
    import diagnostics_bundle

    extra = {"version": _version(), "filevault": _filevault_on(), "speed": _speed(),
             "data_root_synced": bool(_sync_warning(DATA_ROOT))}
    path = diagnostics_bundle.save(DATA_ROOT, extra)
    subprocess.run(["open", "-R", str(path)], check=False)
    return {"ok": True, "path": str(path), "name": path.name}


@app.post("/api/diagnostics/reveal")
async def reveal_logs():
    """Show the log folder in Finder so it can be attached to an email."""
    subprocess.run(["open", str(crash_diagnostics.LOG_DIR)], check=False)
    return {"ok": True}


@app.get("/api/status/{batch_id}")
async def get_status(batch_id: str):
    with _lock:
        b = _batches.get(batch_id)
        if b is None:
            return {"status": "UNKNOWN"}
        return {
            "status": b["status"],
            "log": "\n".join(b["log"][-40:]),
            "total": b["total"],
            "done": b["done"],
            "current": b["current"],
            "cancelled": b["cancelled"],
            "stage": b["stage"],
            "stage_fraction": b["stage_fraction"],
            "stage_detail": b["stage_detail"],
            "csv_path": b.get("csv_path"),
            "files": [
                {
                    "filename": f["filename"],
                    "state": f["state"],
                    "result": f["result"],
                    "warnings": f["warnings"],
                    "error": f["error"],
                    "transcript": f["transcript"],
                    "structured_transcript": f["structured_transcript"],
                    "mask_legend": mask_legend(f["structured_transcript"]),
                    "speaker_roles": f.get("speaker_roles", {}),
                    "speaker_names": f.get("speaker_names", {}),
                    "speaker_samples": f.get("speaker_samples", {}),
                    "preview": f.get("preview"),
                    "clinical_review": (f.get("analysis") or {}).get("clinical_review") or {},
                    "quality": f.get("quality", {}),
                    "score_reliability": f.get("score_reliability", {}),
                }
                for f in b["files"]
            ],
        }


@app.get("/api/analysis/{batch_id}/{index}")
async def get_analysis(batch_id: str, index: int):
    """Full analysis JSON for one file, for the in-app JSON view."""
    with _lock:
        b = _batches.get(batch_id)
        if b is None or index >= len(b["files"]):
            return JSONResponse(status_code=404, content={"error": "not found"})
        return b["files"][index]["analysis"] or {}


def _mmss(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60:02d}:{rest % 60:02d}"


def _batch_markdown(b: dict) -> str:
    """Human-readable report of a whole batch."""
    lines = ["# ClinicalWhisper Analysis", ""]
    for f in b["files"]:
        lines.append(f"## {f['filename']}")
        if f["state"] == "error":
            lines += [f"**Failed:** {f['error']}", ""]
            continue
        for w in f["warnings"]:
            lines.append(f"> Warning: {w}")
        r = f["result"] or {}
        lines += ["", "### Scores", ""]
        for k, v in r.items():
            if k != "filename":
                lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        analysis = f.get("analysis") or {}
        impression = (analysis.get("llm_clinical_scoring") or {}).get("clinical_impression")
        if impression:
            lines += ["", "### Clinical Impression", "", impression]
        observations = (analysis.get("llm_clinical_scoring") or {}).get("key_observations") or []
        if observations:
            lines += ["", "### Key Observations", ""]
            lines += [f"- {o}" for o in observations]
        review = analysis.get("clinical_review") or {}
        if review:
            lines += ["", "### For clinician review", "", f"_{review.get('note', '')}_", ""]
            for it in review.get("items", []):
                lines.append(f"- **{_mmss(it['start'])} · {it['label']}** ({it['role']}): "
                             f"{it['text']}")
        if f["structured_transcript"]:
            key = mask_legend(f["structured_transcript"])
            lines += ["", "### Transcript (de-identified)", ""]
            if key:
                lines += [key.replace("\n  ", "\n- "), ""]
            lines += ["```", f["structured_transcript"], "```"]
        lines.append("")
    return "\n".join(lines)


@app.post("/api/cancel/{batch_id}")
async def cancel_batch(batch_id: str):
    """Ask a running batch to stop at the next checkpoint."""
    with _lock:
        b = _batches.get(batch_id)
        if b is None:
            return JSONResponse(status_code=404, content={"status": "error",
                                                          "message": "Unknown batch."})
        b["cancelled"] = True
        b["log"].append("Cancellation requested...")
    return {"status": "cancelling"}


def _normalise_roles(chosen: dict, segments: list[dict]) -> dict:
    """Roles picked in the window, checked and with "Other" numbered.

    Participant measures are taken from the single Subject, so two Subjects
    are refused rather than one silently winning.
    """
    talk: dict[str, float] = {}
    for seg in segments:
        spk = seg.get("speaker")
        talk[spk] = talk.get(spk, 0.0) + max(seg.get("end", 0.0) - seg.get("start", 0.0), 0.0)
    roles: dict[str, str] = {}
    for spk in sorted(talk, key=lambda s: -talk[s]):
        role = str(chosen.get(spk) or "Other")
        if role.startswith("Other"):
            role = "Other"
        if role not in {"Interviewer", "Subject", "Other"}:
            raise ValueError(f"Unknown role {role!r}.")
        roles[spk] = role
    if sum(r == "Subject" for r in roles.values()) > 1:
        raise ValueError("Only one speaker can be the participant.")
    n = 0
    for spk, role in roles.items():
        if role == "Other":
            n += 1
            roles[spk] = f"Other_{n}"
    return roles


def _roles_with_interviewer(segments: list[dict], interviewer: str) -> dict:
    """The chosen speaker as Interviewer, the other main speaker as Subject."""
    talk: dict[str, float] = {}
    for seg in segments:
        spk = seg.get("speaker")
        talk[spk] = talk.get(spk, 0.0) + max(seg.get("end", 0.0) - seg.get("start", 0.0), 0.0)
    others = sorted((s for s in talk if s != interviewer), key=lambda s: -talk[s])
    roles = {interviewer: "Interviewer"}
    if others:
        roles[others[0]] = "Subject"
    for i, spk in enumerate(others[1:], start=1):
        roles[spk] = f"Other_{i}"
    return roles


@app.post("/api/rescore/{batch_id}/{index}")
async def rescore(batch_id: str, index: int, payload: dict = Body(default={})):
    """Re-run clinical scoring on an already-transcribed file.

    Transcription is the expensive stage, so correcting a speaker role or
    changing the scoring scope should not cost another full pass. Accepts an
    optional ``roles`` mapping to override the detected Interviewer/Subject
    assignment.
    """
    with _lock:
        b = _batches.get(batch_id)
        if b is None or index >= len(b["files"]):
            return JSONResponse(status_code=404, content={"status": "error",
                                                          "message": "Unknown file."})
        f = b["files"][index]
        segments = f.get("segments") or []
        analysis = f.get("analysis") or {}

    if not segments:
        return JSONResponse(status_code=400,
                            content={"status": "error",
                                     "message": "No cached transcript for this file."})

    try:
        from acoustic_context import build_acoustic_prompt_context
        from llm_clinical_scorer import score_transcript
        from transcript_formatter import (clean_names, compute_speaker_stats,
                                          format_structured_transcript)

        cfg = load_config(_config_path())
        old_roles = analysis.get("speaker_roles") or {}
        roles = _normalise_roles(payload["roles"], segments) if payload.get("roles") else old_roles
        if payload.get("interviewer"):
            roles = _roles_with_interviewer(segments, payload["interviewer"])
        speakers = {seg.get("speaker") for seg in segments}
        names = (clean_names(payload["names"], speakers) if "names" in payload
                 else analysis.get("speaker_names") or {})
        with _lock:
            transcribe_only = bool((_batches.get(batch_id) or {}).get("transcribe_only"))
        scope = payload.get("transcript_scope")
        if scope:
            cfg.setdefault("llm_scoring", {})["transcript_scope"] = scope

        structured = format_structured_transcript(segments, roles, names)
        context = build_acoustic_prompt_context(
            analysis.get("overall_acoustics", {}) or {},
            analysis.get("speaker_acoustics", {}) or {},
        )
        # A transcribe-only batch, or a change of names only, just gets
        # relabelled; a role change re-scores, off the event loop so the window
        # keeps updating during a long re-score.
        rescored = not transcribe_only and roles != old_roles
        if rescored:
            scoring = await run_in_threadpool(score_transcript, structured, context, cfg)
        else:
            scoring = analysis.get("llm_clinical_scoring") or {}

        analysis = dict(analysis)
        analysis["llm_clinical_scoring"] = scoring
        analysis["speaker_roles"] = roles
        analysis["speaker_names"] = names
        analysis["structured_transcript"] = structured
        analysis["speaker_stats"] = compute_speaker_stats(segments, roles)
        # Who counts as the interviewer decides whose lines are screened.
        import review_flags
        analysis["clinical_review"] = review_flags.find(
            segments, roles, (cfg.get("review_flags") or {}).get("extra_terms"))

        # A role swap changes who the participant is: participant timing and
        # the quality flags follow, and reliability follows the run count.
        from clinical_safeguards import assess_quality, score_reliability
        from timing_features import subject_speaker

        timing = dict(analysis.get("timing_features") or {})
        timing["subject_speaker"] = subject_speaker(timing.get("per_speaker") or {}, roles)
        analysis["timing_features"] = timing
        previous = analysis.get("quality") or {}
        analysis["quality"] = assess_quality(
            segments, timing["subject_speaker"],
            {k: previous.get(k) for k in ("duration_s", "rms_dbfs", "clipped_fraction", "snr_db")},
            expected_speakers=cfg.get("moss", {}).get("num_speakers"),
        )
        runs = (scoring.get("_meta") or {}).get("samples_per_window") or \
            cfg.get("llm_scoring", {}).get("samples", 1)
        analysis["score_reliability"] = score_reliability(runs)

        path = f.get("analysis_path")
        if path:
            Path(path).write_text(json.dumps(analysis, indent=2, default=str),
                                  encoding="utf-8")

        from batch_processor import _extract_row
        row = _extract_row(path, f["filename"]) if path else f.get("result")

        with _lock:
            tgt = _batches[batch_id]["files"][index]
            tgt["analysis"] = analysis
            tgt["result"] = row
            tgt["structured_transcript"] = structured
            tgt["speaker_roles"] = roles
            tgt["speaker_names"] = names
            tgt["quality"] = analysis.get("quality") or {}
            tgt["score_reliability"] = analysis.get("score_reliability") or {}
            _batches[batch_id]["log"].append(
                f"Re-scored {f['filename']}." if rescored else f"Updated speakers for {f['filename']}.")

        return {"status": "success", "result": row, "rescored": rescored,
                "quality": analysis.get("quality"),
                "score_reliability": analysis.get("score_reliability"),
                "structured_transcript": structured, "speaker_roles": roles,
                "speaker_names": names,
                "clinical_review": analysis.get("clinical_review") or {}}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        log.warning("Re-score failed for %s: %s", batch_id, e)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/save/{batch_id}/{fmt}")
async def save_output(batch_id: str, fmt: str):
    """Write the batch results as csv | json | md, wherever the user picks."""
    if fmt not in {"csv", "json", "md"}:
        return JSONResponse(status_code=400, content={"status": "error",
                                                      "message": f"Unknown format {fmt}"})
    with _lock:
        b = _batches.get(batch_id)
        if b is None or b["done"] == 0:
            return JSONResponse(status_code=404,
                                content={"status": "error",
                                         "message": "No results available to save."})
        csv_source = b.get("csv_path")
        payload_json = json.dumps(
            [f["analysis"] for f in b["files"] if f["analysis"]],
            indent=2, default=str,
        )
        payload_md = _batch_markdown(b)
        default_name = f"clinicalwhisper_results.{fmt}"

    if sys.platform != "darwin":
        target = OUTPUT_DIR / default_name
        if fmt == "csv" and csv_source:
            shutil.copy2(csv_source, target)
        else:
            target.write_text(payload_json if fmt == "json" else payload_md, encoding="utf-8")
        return {"status": "success", "path": str(target)}

    try:
        apple_script = (
            f'set saveFile to choose file name with prompt "Save results as:" '
            f'default name "{default_name}"\n'
            f"POSIX path of saveFile"
        )
        result = subprocess.run(["osascript", "-e", apple_script],
                                capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return {"status": "cancelled"}

        target = Path(result.stdout.strip())
        if fmt == "csv":
            if not csv_source or not Path(csv_source).exists():
                return JSONResponse(status_code=404,
                                    content={"status": "error",
                                             "message": "No CSV was produced for this batch."})
            shutil.copy2(csv_source, target)
        else:
            target.write_text(payload_json if fmt == "json" else payload_md, encoding="utf-8")

        # Saved, but say so if the chosen folder uploads what is put in it.
        return {"status": "success", "path": str(target), "warning": _sync_warning(target)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


app.mount("/", StaticFiles(directory=WWW_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("gui_server:app", host="127.0.0.1", port=8000)
