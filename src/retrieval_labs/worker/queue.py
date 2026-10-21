"""The analysis queue: one RQ job per processing job, never retried (D-003)."""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

from rq import Queue

QUEUE_NAME = "analysis"
TASK = "retrieval_labs.worker.tasks.run_job"
# A job still RUNNING this long after its time limit lost its worker (crash, kill, restart).
LOST_JOB_GRACE = timedelta(minutes=2)


def job_timeout_s() -> int:
    """Per-job time limit, from ``JOB_TIMEOUT_S`` (default 15 minutes)."""
    return int(os.environ.get("JOB_TIMEOUT_S", "900"))


def lost_after() -> timedelta:
    return timedelta(seconds=job_timeout_s()) + LOST_JOB_GRACE


def enqueue_job(queue: Queue, job_id: uuid.UUID) -> None:
    # The RQ job id is the processing job id, so a re-enqueue replaces rather than duplicates;
    # process_job's PENDING claim makes any second run a no-op anyway.
    queue.enqueue(
        TASK,
        str(job_id),
        job_id=str(job_id),
        job_timeout=job_timeout_s(),
        retry=None,
        result_ttl=24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
