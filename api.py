#!/usr/bin/env python3
"""FastAPI ingestion layer for ClinicalWhisper async jobs."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cw_config import load_config, resolve_path
from queue_store import JobQueue

CFG = load_config()
PIPELINE_CFG = CFG.get("pipeline", {})
QUEUE = JobQueue(resolve_path(PIPELINE_CFG.get("queue_db_path", "./clinicalwhisper_jobs.db")))
SECURE_STORAGE_DIR = Path(
    resolve_path(PIPELINE_CFG.get("secure_storage_folder", CFG.get("input_folder", "./Input")))
)
SECURE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_EXTENSIONS = tuple(ext.lower() for ext in CFG.get("audio_extensions", []))

app = FastAPI(
    title="ClinicalWhisper Ingestion API",
    version="3.0",
    description="Upload audio and receive a non-blocking job_id for async processing.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    """Ingest audio, persist it immediately, enqueue a job, and return job_id."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(AUDIO_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    stored_path = SECURE_STORAGE_DIR / f"{job_id}{ext}"

    with stored_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    await file.close()
    os.chmod(stored_path, 0o600)

    QUEUE.enqueue_job(job_id=job_id, file_path=str(stored_path), original_filename=safe_name)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = QUEUE.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


@app.get("/jobs")
def list_jobs(limit: int = 25) -> list[dict]:
    limit = max(1, min(limit, 200))
    return QUEUE.list_jobs(limit=limit)


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """
    Return completed analysis payload.

    This lets downstream clients fetch the exact JSON saved for app.py.
    """
    job = QUEUE.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is '{job['status']}'")

    output_json = job.get("output_json_path")
    if not output_json:
        raise HTTPException(status_code=500, detail="Completed job missing output path")

    out_path = Path(output_json)
    if not out_path.exists():
        raise HTTPException(status_code=500, detail="Output JSON not found on disk")

    with out_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return JSONResponse(content=payload)
