"""Queue and worker against real Redis and PostgreSQL, with free stand-in models."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest
from rq import Queue
from rq.job import Job, JobStatus
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.db import repository
from retrieval_labs.db.models import AnalysisResult, Document, JobState, ProcessingJob
from retrieval_labs.extraction.llm import FakeExtractor
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.worker.main import AnalysisWorker
from retrieval_labs.worker.queue import enqueue_job
from retrieval_labs.worker.tasks import Runtime, configure

pytestmark = pytest.mark.integration

TIMEOUT_S = 900
TEXT = "This Agreement shall be governed by the laws of the State of New York."


@pytest.fixture
def runtime(sessions: sessionmaker[Session]) -> Iterator[Runtime]:
    runtime = Runtime(sessions=sessions, retriever=BM25Retriever(), extractor=FakeExtractor())
    configure(runtime)
    yield runtime
    configure(None)


def submit(sessions: sessionmaker[Session]) -> uuid.UUID:
    with sessions.begin() as session:
        document = repository.create_document(
            session, title="Agreement", filename="a.txt", text=TEXT
        )
        return repository.create_job(session, document.id, retriever="bm25", llm_provider="fake").id


def drain(queue: Queue) -> None:
    AnalysisWorker([queue], connection=queue.connection).work(burst=True, with_scheduler=False)


def test_a_queued_job_is_completed_by_the_worker(runtime: Runtime, queue: Queue) -> None:
    job_id = submit(runtime.sessions)
    enqueue_job(queue, job_id, TIMEOUT_S)
    drain(queue)

    rq_job = Job.fetch(str(job_id), connection=queue.connection)
    assert rq_job.get_status() == JobStatus.FINISHED
    assert rq_job.return_value() == "COMPLETED"
    with runtime.sessions() as session:
        assert repository.get_job(session, job_id).state is JobState.COMPLETED
        assert repository.get_report(session, job_id) is not None


def test_queued_jobs_have_no_retries_and_a_time_limit(queue: Queue) -> None:
    job_id = uuid.uuid4()
    enqueue_job(queue, job_id, 123)
    rq_job = Job.fetch(str(job_id), connection=queue.connection)
    assert rq_job.timeout == 123
    assert not rq_job.retries_left
    assert rq_job.func_name == "retrieval_labs.worker.tasks.run_job"


def test_enqueueing_twice_runs_the_pipeline_once(runtime: Runtime, queue: Queue) -> None:
    job_id = submit(runtime.sessions)
    enqueue_job(queue, job_id, TIMEOUT_S)
    enqueue_job(queue, job_id, TIMEOUT_S)
    drain(queue)
    calls = len(runtime.extractor.requests)
    enqueue_job(queue, job_id, TIMEOUT_S)
    drain(queue)
    assert len(runtime.extractor.requests) == calls
    with runtime.sessions() as session:
        assert len(session.scalars(select(AnalysisResult)).all()) == 1


def test_the_worker_marks_long_running_jobs_lost_and_does_not_rerun_them(
    runtime: Runtime, queue: Queue
) -> None:
    lost_id = submit(runtime.sessions)
    recent_id = submit(runtime.sessions)
    with runtime.sessions.begin() as session:
        for job_id, age in ((lost_id, timedelta(hours=2)), (recent_id, timedelta(0))):
            session.execute(
                update(ProcessingJob)
                .where(ProcessingJob.id == job_id)
                .values(state=JobState.RUNNING, started_at=ProcessingJob.created_at - age)
            )
    next_id = submit(runtime.sessions)
    enqueue_job(queue, next_id, TIMEOUT_S)
    drain(queue)

    with runtime.sessions() as session:
        lost = repository.get_job(session, lost_id)
        recent = repository.get_job(session, recent_id)
        assert lost.state is JobState.FAILED
        assert lost.error_code == "worker_lost"
        assert recent.state is JobState.RUNNING
        assert repository.get_job(session, next_id).state is JobState.COMPLETED
        assert repository.get_report(session, lost_id) is None  # not re-run
        assert len(session.scalars(select(Document)).all()) == 3  # nothing deleted


class SlowExtractor(FakeExtractor):
    def extract(self, request):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:  # short sleeps, so the timeout can interrupt
            time.sleep(0.01)
        return super().extract(request)


def test_a_job_over_its_time_limit_fails_with_timeout(
    sessions: sessionmaker[Session], queue: Queue
) -> None:
    configure(Runtime(sessions=sessions, retriever=BM25Retriever(), extractor=SlowExtractor()))
    try:
        job_id = submit(sessions)
        enqueue_job(queue, job_id, 1)
        started = time.monotonic()
        drain(queue)
        assert time.monotonic() - started < 4
    finally:
        configure(None)
    with sessions() as session:
        job = repository.get_job(session, job_id)
    assert job.state is JobState.FAILED
    assert job.error_code == "timeout"
