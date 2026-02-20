#!/usr/bin/env python3
"""Background worker runner for ClinicalWhisper async inference."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from typing import Optional

from cw_config import load_config, resolve_path
from inference_pipeline import InferencePipeline
from queue_store import JobQueue


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def worker_loop(worker_id: str, config_path: Optional[str], stop_event) -> None:
    configure_logging()
    log = logging.getLogger(f"ClinicalWhisper[{worker_id}]")

    cfg = load_config(config_path)
    pipeline_cfg = cfg.get("pipeline", {})
    queue = JobQueue(resolve_path(pipeline_cfg.get("queue_db_path", "./clinicalwhisper_jobs.db")))
    poll_interval = float(pipeline_cfg.get("poll_interval_seconds", 1.0))

    log.info("Starting worker loop (poll_interval=%.2fs)", poll_interval)
    pipeline = InferencePipeline(cfg)

    while not stop_event.is_set():
        job = queue.claim_next_job(worker_id)
        if not job:
            time.sleep(poll_interval)
            continue

        job_id = job["job_id"]
        log.info("Claimed job %s", job_id)

        try:
            output_json = pipeline.process_job(job)
            queue.mark_completed(job_id, output_json)
            log.info("Job %s completed", job_id)
        except Exception as exc:
            queue.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
            log.exception("Job %s failed", job_id)

    log.info("Stop signal received, exiting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClinicalWhisper background workers")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to config.pipeline.max_workers)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    default_workers = int(cfg.get("pipeline", {}).get("max_workers", 2))
    num_workers = args.workers if args.workers and args.workers > 0 else default_workers

    configure_logging()
    log = logging.getLogger("ClinicalWhisperWorker")
    log.info("Launching %d worker process(es)", num_workers)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    processes: list[mp.Process] = []

    for idx in range(num_workers):
        worker_id = f"worker-{idx + 1}"
        process = ctx.Process(
            target=worker_loop,
            args=(worker_id, args.config, stop_event),
            name=worker_id,
        )
        process.start()
        processes.append(process)

    try:
        while any(p.is_alive() for p in processes):
            for process in processes:
                process.join(timeout=0.5)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, stopping workers...")
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    finally:
        log.info("All workers stopped")


if __name__ == "__main__":
    main()
