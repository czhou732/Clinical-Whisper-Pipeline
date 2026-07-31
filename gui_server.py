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
import uuid
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import bundled_models

# Harmless when running from source; redirects to the in-bundle weights when
# frozen. Runs before the pipeline imports transformers.
bundled_models.configure()

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

    try:
        # Imported here, inside the try: a missing dependency in a frozen build
        # would otherwise raise before any state is set, leaving the UI stuck.
        from batch_processor import _extract_row
        from inference_pipeline import InferencePipeline

        cfg = load_config(_config_path())

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

        rows: list[dict] = []
        for idx, audio_path in enumerate(paths):
            if _cancelled():
                with _lock:
                    for f in _batches[batch_id]["files"]:
                        if f["state"] in ("pending", "running"):
                            f["state"] = "cancelled"
                    _batches[batch_id]["log"].append("Cancelled by user.")
                break

            with _lock:
                b = _batches[batch_id]
                b["current"] = audio_path.name
                b["files"][idx]["state"] = "running"
                b["log"].append(f"[{idx + 1}/{len(paths)}] {audio_path.name}")

            try:
                with _lock:
                    meta = (
                        _batches[batch_id].get("participant_id", ""),
                        _batches[batch_id].get("session_label", ""),
                    )
                job = {
                    "job_id": f"gui_{uuid.uuid4().hex[:12]}",
                    "file_path": str(audio_path),
                    "original_filename": audio_path.name,
                    "participant_id": meta[0],
                    "session_label": meta[1],
                }
                analysis_path = pipeline.process_job(job)
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
                    # Kept so roles can be corrected and the file re-scored
                    # without paying for transcription again.
                    f["segments"] = analysis.get("segments", [])
                    f["speaker_roles"] = analysis.get("speaker_roles", {})
                    _batches[batch_id]["done"] += 1
            except Exception as exc:
                # A user-requested stop surfaces here as an exception too, but
                # it is not a failure and must not be reported as one.
                if _cancelled():
                    with _lock:
                        _batches[batch_id]["files"][idx]["state"] = "cancelled"
                        _batches[batch_id]["log"].append(
                            f"Stopped during {audio_path.name}."
                        )
                    break

                # One bad file must not abort the rest of the batch.
                log.warning("Batch %s: %s failed: %s", batch_id, audio_path.name, exc)
                with _lock:
                    f = _batches[batch_id]["files"][idx]
                    f["state"] = "error"
                    f["error"] = str(exc) or exc.__class__.__name__
                    _batches[batch_id]["done"] += 1
                    _batches[batch_id]["log"].append(f"ERROR ({audio_path.name}): {exc}")

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
        cw_log.removeHandler(handler)


# ── API ─────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    participant_id: str = Form(""),
    session_label: str = Form(""),
):
    """Accept one or many audio files and start a single batch job."""
    try:
        saved: list[Path] = []
        for f in files:
            safe_name = Path(f.filename or "audio").name
            dest = INPUT_DIR / safe_name
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved.append(dest)

        if not saved:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No files received."},
            )

        batch_id = _new_batch(saved)
        with _lock:
            _batches[batch_id]["participant_id"] = participant_id.strip()
            _batches[batch_id]["session_label"] = session_label.strip()

        # An explicit daemon thread rather than FastAPI BackgroundTasks: a batch
        # runs for minutes, which would pin an anyio threadpool slot for its
        # whole duration, and in the frozen app the task was observed never
        # being dispatched at all — leaving the UI stuck on "Queued".
        threading.Thread(
            target=process_batch_task,
            args=(batch_id, saved),
            name=f"cw-{batch_id}",
            daemon=True,
        ).start()

        return {
            "status": "success",
            "batch_id": batch_id,
            "count": len(saved),
            "filenames": [p.name for p in saved],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


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
                    "speaker_roles": f.get("speaker_roles", {}),
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
        if f["structured_transcript"]:
            lines += ["", "### Transcript (de-identified)", "", "```",
                      f["structured_transcript"], "```"]
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
        from transcript_formatter import format_structured_transcript, compute_speaker_stats

        cfg = load_config(_config_path())
        roles = payload.get("roles") or analysis.get("speaker_roles") or {}
        scope = payload.get("transcript_scope")
        if scope:
            cfg.setdefault("llm_scoring", {})["transcript_scope"] = scope

        structured = format_structured_transcript(segments, roles)
        context = build_acoustic_prompt_context(
            analysis.get("overall_acoustics", {}) or {},
            analysis.get("speaker_acoustics", {}) or {},
        )
        scoring = score_transcript(structured, context, cfg)

        analysis = dict(analysis)
        analysis["llm_clinical_scoring"] = scoring
        analysis["speaker_roles"] = roles
        analysis["structured_transcript"] = structured
        analysis["speaker_stats"] = compute_speaker_stats(segments, roles)

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
            _batches[batch_id]["log"].append(f"Re-scored {f['filename']}.")

        return {"status": "success", "result": row,
                "structured_transcript": structured, "speaker_roles": roles}
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

        return {"status": "success", "path": str(target)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


app.mount("/", StaticFiles(directory=WWW_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uvicorn.run("gui_server:app", host="127.0.0.1", port=8000)
