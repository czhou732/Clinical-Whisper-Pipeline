"""FastAPI backend for the ClinicalWhisper desktop GUI.

Serves the static frontend from the app bundle and runs the inference pipeline
on a single uploaded file, streaming stage-level log lines back to the UI.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cw_config import DATA_ROOT, load_config

app = FastAPI(title="ClinicalWhisper GUI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Code and static assets live in the (read-only) bundle; user data does not.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
WWW_DIR = BUNDLE_DIR / "www"

INPUT_DIR = DATA_ROOT / "Input"
OUTPUT_DIR = DATA_ROOT / "Output"
for _d in (INPUT_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _config_path() -> str:
    """Prefer a user-editable config, fall back to the bundled example."""
    user_cfg = DATA_ROOT / "config.yaml"
    if user_cfg.exists():
        return str(user_cfg)
    return str(BUNDLE_DIR / "config.example.yaml")


# ── Task state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
tasks_state: dict[str, str] = {}
tasks_results: dict[str, dict] = {}
tasks_logs: dict[str, list[str]] = {}
tasks_csv: dict[str, str] = {}
tasks_warnings: dict[str, list[str]] = {}


class _TaskLogHandler(logging.Handler):
    """Mirrors pipeline log records into the task's log buffer for the GUI."""

    def __init__(self, task_id: str):
        super().__init__(level=logging.INFO)
        self.task_id = task_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = record.getMessage()
        except Exception:
            return
        with _lock:
            buf = tasks_logs.setdefault(self.task_id, [])
            buf.append(line)
            del buf[:-200]  # keep the tail bounded
            tasks_state[self.task_id] = line


def process_audio_task(task_id: str, audio_path: Path) -> None:
    """Run the full pipeline on exactly one file."""
    handler = _TaskLogHandler(task_id)
    cw_log = logging.getLogger("ClinicalWhisper")
    cw_log.addHandler(handler)
    cw_log.setLevel(logging.INFO)

    with _lock:
        tasks_state[task_id] = "Loading models..."
        tasks_logs[task_id] = ["Starting pipeline..."]

    try:
        # Imported here, inside the try: a missing dependency in a frozen build
        # would otherwise raise before the state is set, leaving the UI stuck on
        # "Queued" forever with nothing to diagnose.
        from batch_processor import _extract_row
        from inference_pipeline import InferencePipeline

        cfg = load_config(_config_path())
        pipeline = InferencePipeline(cfg)
        job = {
            "job_id": f"gui_{uuid.uuid4().hex[:12]}",
            "file_path": str(audio_path),
            "original_filename": audio_path.name,
        }
        analysis_path = pipeline.process_job(job)
        row = _extract_row(analysis_path, audio_path.name)

        # A run where a stage failed still produces a CSV, but with blank
        # columns. Pass the warnings through so the UI can say so.
        with open(analysis_path, "r", encoding="utf-8") as fh:
            warnings = json.load(fh).get("warnings", [])

        csv_path = OUTPUT_DIR / f"{audio_path.stem}_summary.csv"
        pd.DataFrame([row]).to_csv(csv_path, index=False)

        with _lock:
            tasks_results[task_id] = row
            tasks_csv[task_id] = str(csv_path)
            tasks_warnings[task_id] = warnings
            tasks_state[task_id] = "COMPLETED"
            for w in warnings:
                tasks_logs[task_id].append(f"WARNING: {w}")
            tasks_logs[task_id].append(f"Analysis written to {analysis_path}")
    except Exception as e:
        with _lock:
            tasks_state[task_id] = f"ERROR: {e}"
            tasks_logs.setdefault(task_id, []).append(f"ERROR: {e}")
    finally:
        cw_log.removeHandler(handler)


# ── API ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        safe_name = Path(file.filename or "audio").name
        file_path = INPUT_DIR / safe_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task_id = f"task_{uuid.uuid4().hex[:10]}"
        with _lock:
            tasks_state[task_id] = "Queued"
            tasks_logs[task_id] = []

        # An explicit daemon thread rather than FastAPI BackgroundTasks: a job
        # runs for minutes, which would pin an anyio threadpool slot for its
        # whole duration, and in the frozen app the task was observed never
        # being dispatched at all — leaving the UI stuck on "Queued".
        worker = threading.Thread(
            target=process_audio_task,
            args=(task_id, file_path),
            name=f"cw-{task_id}",
            daemon=True,
        )
        worker.start()

        return {"status": "success", "task_id": task_id, "filename": safe_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    with _lock:
        res = {
            "status": tasks_state.get(task_id, "UNKNOWN"),
            "log": "\n".join(tasks_logs.get(task_id, [])[-40:]),
        }
        if res["status"] == "COMPLETED" and task_id in tasks_results:
            res["result"] = tasks_results[task_id]
            res["csv_path"] = tasks_csv.get(task_id)
            res["warnings"] = tasks_warnings.get(task_id, [])
    return res


@app.post("/api/save_output/{task_id}")
async def save_output(task_id: str):
    """Copy this run's CSV wherever the user picks, via a native save dialog."""
    with _lock:
        source_csv = tasks_csv.get(task_id)

    if not source_csv or not Path(source_csv).exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No results available to save."},
        )

    if sys.platform != "darwin":
        return {"status": "success", "path": source_csv, "note": "saved in place"}

    try:
        default_name = Path(source_csv).name
        apple_script = (
            f'set saveFile to choose file name with prompt "Save results as:" '
            f'default name "{default_name}"\n'
            f"POSIX path of saveFile"
        )
        result = subprocess.run(
            ["osascript", "-e", apple_script], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            target_path = result.stdout.strip()
            shutil.copy2(source_csv, target_path)
            return {"status": "success", "path": target_path}
        return {"status": "cancelled"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


app.mount("/", StaticFiles(directory=WWW_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("gui_server:app", host="127.0.0.1", port=8000)
