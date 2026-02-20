#!/usr/bin/env python3
"""
SQLite-backed job broker for ClinicalWhisper.

This provides a lightweight queue that decouples ingestion from inference
without requiring Redis/RabbitMQ.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

JOB_STATUSES = ("queued", "processing", "completed", "failed")


def utcnow_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    """Persistent queue operations backed by SQLite."""

    def __init__(self, db_path: str):
        db = Path(db_path).expanduser()
        self.db_path = str(db)
        db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    worker_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    output_json_path TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);"
            )

    def enqueue_job(self, job_id: str, file_path: str, original_filename: str) -> None:
        now = utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, original_filename, file_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (job_id, original_filename, file_path, now, now),
            )

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, original_filename, file_path, status, created_at, updated_at,
                       started_at, completed_at, worker_id, attempts, error, output_json_path
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def claim_next_job(self, worker_id: str) -> Optional[dict]:
        """
        Atomically claim the oldest queued job for a worker.

        Uses a transaction so multiple worker processes can safely race.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job_id
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                conn.commit()
                return None

            now = utcnow_iso()
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    worker_id = ?,
                    started_at = ?,
                    updated_at = ?,
                    attempts = attempts + 1
                WHERE job_id = ? AND status = 'queued'
                """,
                (worker_id, now, now, row["job_id"]),
            ).rowcount
            conn.commit()

            if updated != 1:
                return None

            return self.get_job(row["job_id"])
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_completed(self, job_id: str, output_json_path: str) -> None:
        now = utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    output_json_path = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error = NULL
                WHERE job_id = ?
                """,
                (output_json_path, now, now, job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        now = utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (error[:4000], now, now, job_id),
            )

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, original_filename, status, created_at, updated_at, output_json_path
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
