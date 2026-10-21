"""The API end to end: real PostgreSQL and Redis, an in-process worker with free models."""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rq import Queue
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.api.app import api_redis, create_app
from retrieval_labs.config import Settings
from retrieval_labs.db.models import AnalysisResult, Chunk, Document, JobState, ProcessingJob
from retrieval_labs.extraction.llm import FakeExtractor
from retrieval_labs.logging_setup import configure_logging
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.worker.main import AnalysisWorker
from retrieval_labs.worker.tasks import Runtime, configure

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
# CRLF, non-ASCII and trailing spaces: none of it may be normalized away.
TEXT = (
    "MASTER SERVICES AGREEMENT  \r\n\r\n"
    "1. Services. Provider shall perform the services described in Schedule A.\r\n"
    "2. Governing Law. This Agreement is governed by the laws of the State of Delaware.\r\n"
    "3. Notices. Notices go to the address of the Société Générale branch.\r\n"
)


def make_client(sessions: sessionmaker[Session], queue: Queue) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.sessions = sessions
        app.state.queue = queue
        yield

    settings = Settings(database_url="unused", redis_url="unused")
    app = create_app(settings, lifespan=lifespan, reports_dir=ROOT / "reports" / "benchmark")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(sessions: sessionmaker[Session], queue: Queue) -> Iterator[TestClient]:
    configure(Runtime(sessions=sessions, retriever=BM25Retriever(), extractor=FakeExtractor()))
    with make_client(sessions, queue) as client:
        yield client
    configure(None)


def submit(client: TestClient, text: str = TEXT, **form: str) -> dict:
    response = client.post(
        "/contracts",
        files={"file": ("msa.txt", text.encode("utf-8"), "text/plain")},
        data=form,
    )
    assert response.status_code == 202, response.text
    return response.json()


def drain(queue: Queue) -> None:
    AnalysisWorker([queue], connection=queue.connection).work(burst=True, with_scheduler=False)


def test_submit_process_and_read_the_report(
    client: TestClient, queue: Queue, sessions: sessionmaker[Session]
) -> None:
    submission = submit(client, title="Master Services Agreement")
    document_id = submission["document_id"]
    assert submission["state"] == "PENDING"
    assert submission["status_url"] == f"/contracts/{document_id}/status"

    status = client.get(submission["status_url"]).json()
    assert (status["state"], status["job_id"]) == ("PENDING", submission["job_id"])
    response = client.get(submission["report_url"])
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "report_not_ready"

    drain(queue)

    status = client.get(submission["status_url"]).json()
    assert status["state"] == "COMPLETED"
    assert status["error_code"] is None
    assert status["started_at"] and status["finished_at"]
    response = client.get(submission["report_url"])
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["contract_id"] == document_id
    assert body["report"]["title"] == "Master Services Agreement"
    assert len(body["report"]["results"]) == 5

    with sessions() as session:
        stored = session.scalars(select(Document)).one()
        assert stored.text == TEXT  # byte for byte, CRLF and all
        for chunk in session.scalars(select(Chunk)):
            assert 0 <= chunk.start < chunk.end <= len(TEXT)


def test_status_and_report_of_an_unknown_contract_are_404(client: TestClient) -> None:
    for path in ("status", "report"):
        response = client.get(f"/contracts/00000000-0000-0000-0000-000000000000/{path}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_a_failed_job_has_no_report(client: TestClient, sessions: sessionmaker[Session]) -> None:
    submission = submit(client)
    with sessions.begin() as session:
        session.execute(
            update(ProcessingJob).values(state=JobState.FAILED, error_code="processing_error")
        )
    response = client.get(submission["report_url"])
    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "job_failed",
        "message": "the analysis failed (processing_error)",
    }


def test_status_reads_mark_lost_jobs(client: TestClient, sessions: sessionmaker[Session]) -> None:
    submission = submit(client)
    with sessions.begin() as session:
        session.execute(
            update(ProcessingJob).values(
                state=JobState.RUNNING, started_at=ProcessingJob.created_at - timedelta(hours=2)
            )
        )
    status = client.get(submission["status_url"]).json()
    assert (status["state"], status["error_code"]) == ("FAILED", "worker_lost")


def test_delete_removes_the_contract_and_everything_derived(
    client: TestClient, queue: Queue, sessions: sessionmaker[Session]
) -> None:
    submission = submit(client)
    drain(queue)
    response = client.delete(f"/contracts/{submission['document_id']}")
    assert response.status_code == 204
    assert client.get(submission["status_url"]).status_code == 404
    with sessions() as session:
        for model in (Document, Chunk, ProcessingJob, AnalysisResult):
            assert session.scalars(select(model)).all() == []
    assert client.delete(f"/contracts/{submission['document_id']}").status_code == 404


def test_a_running_job_cannot_be_deleted(
    client: TestClient, sessions: sessionmaker[Session]
) -> None:
    submission = submit(client)
    with sessions.begin() as session:
        session.execute(update(ProcessingJob).values(state=JobState.RUNNING))
    response = client.delete(f"/contracts/{submission['document_id']}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_running"


def test_a_contract_deleted_while_queued_is_never_processed(
    client: TestClient, queue: Queue
) -> None:
    submission = submit(client)
    assert client.delete(f"/contracts/{submission['document_id']}").status_code == 204
    drain(queue)  # the queued job finds nothing to claim


def test_unavailable_queue_fails_the_job_and_returns_503(
    sessions: sessionmaker[Session],
) -> None:
    unreachable = Queue("analysis-unreachable", connection=api_redis("redis://127.0.0.1:1/0"))
    with make_client(sessions, unreachable) as client:
        started = time.monotonic()
        response = client.post(
            "/contracts", files={"file": ("a.txt", TEXT.encode("utf-8"), "text/plain")}
        )
        assert time.monotonic() - started < 10  # fails fast, no long retry loop
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "queue_unavailable"
    with sessions() as session:
        job = session.scalars(select(ProcessingJob)).one()
        assert (job.state, job.error_code) == (JobState.FAILED, "queue_unavailable")


def test_health_is_ok_with_both_services(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "database": "ok", "queue": "ok"}


def test_contract_text_is_never_logged(
    client: TestClient, queue: Queue, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        submission = submit(client)
        drain(queue)
        client.get(submission["report_url"])
    assert caplog.records
    for record in caplog.records:
        logged = record.getMessage() + str(record.__dict__)
        assert "Schedule A" not in logged and "Delaware" not in logged


def test_a_full_job_writes_json_logs_without_contract_text(
    client: TestClient, queue: Queue
) -> None:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    buffer = io.StringIO()
    configure_logging("DEBUG", buffer)
    try:
        submission = submit(client)
        drain(queue)
        client.get(submission["report_url"])
        client.post("/contracts", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])

    entries = [json.loads(line) for line in buffer.getvalue().split("\n") if line]
    events = {entry["event"] for entry in entries}
    assert {"contract submitted", "job completed"} <= events
    submitted = next(e for e in entries if e["event"] == "contract submitted")
    assert submitted["document_id"] == submission["document_id"]
    assert submitted["characters"] == len(TEXT)
    assert len(submitted["text_sha256"]) == 64
    for phrase in ("Schedule A", "Delaware", "Société", "Soci\u00e9t\u00e9"):
        assert phrase not in buffer.getvalue()
