"""The analysis queue: one RQ job per processing job, never retried (D-003)."""

from __future__ import annotations

import uuid

from rq import Queue

QUEUE_NAME = "analysis"
TASK = "retrieval_labs.worker.tasks.run_job"


def enqueue_job(queue: Queue, job_id: uuid.UUID, timeout_s: int) -> None:
    # The RQ job id is the processing job id, so a re-enqueue replaces rather than duplicates;
    # process_job's PENDING claim makes any second run a no-op anyway.
    queue.enqueue(
        TASK,
        str(job_id),
        job_id=str(job_id),
        job_timeout=timeout_s,
        retry=None,
        result_ttl=24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
