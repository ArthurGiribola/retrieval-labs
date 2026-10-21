"""FastAPI application. Submitting a contract stores it, queues a job and returns at once; the
worker (``retrieval_labs.worker.main``) does the analysis.

    uv run python -m retrieval_labs.api      (settings from the environment, JSON logs)

No authentication or rate limiting: the API must only listen on the loopback interface
(brief Section 10.1; Compose publishes it on 127.0.0.1 only).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from rq import Queue
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.api.errors import (
    ApiError,
    ErrorResponse,
    error_response,
    install_error_handlers,
)
from retrieval_labs.api.uploads import (
    check_file_type,
    clean_title,
    decode_contract,
    safe_filename,
)
from retrieval_labs.config import Settings, load_settings
from retrieval_labs.db import repository
from retrieval_labs.db.models import JobState, ProcessingJob
from retrieval_labs.db.session import make_engine, make_session_factory
from retrieval_labs.evaluation.report_tables import REPORTS_DIR, TEST_RUN, BenchmarkRow, load_rows
from retrieval_labs.pipeline import ContractReport
from retrieval_labs.worker.queue import QUEUE_NAME, enqueue_job

logger = logging.getLogger(__name__)

# Room for multipart boundaries and form fields on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
ERRORS = {"model": ErrorResponse}


class Submission(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    state: JobState
    status_url: str
    report_url: str


class JobStatus(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    state: JobState
    error_code: str | None
    retriever: str
    llm_provider: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class Report(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    report: ContractReport


class Benchmark(BaseModel):
    run: str
    split: str
    note: str
    rows: list[BenchmarkRow]


class Health(BaseModel):
    status: str
    database: str
    queue: str


def api_redis(url: str) -> Redis:
    """Fail fast: redis-py 8 retries 10 times with backoff by default, which held a request
    for 49 s when Redis was down. The API answers 503 instead; the worker keeps the default."""
    return Redis.from_url(
        url, socket_connect_timeout=2, socket_timeout=5, retry=Retry(NoBackoff(), 1)
    )


@asynccontextmanager
async def default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = getattr(app.state, "settings", None) or load_settings()
    app.state.settings = settings
    engine = make_engine(settings.database_url)
    redis = api_redis(settings.redis_url)
    app.state.sessions = make_session_factory(engine)
    app.state.queue = Queue(QUEUE_NAME, connection=redis)
    yield
    redis.close()
    engine.dispose()


def get_session(request: Request) -> Iterator[Session]:
    sessions: sessionmaker[Session] = request.app.state.sessions
    with sessions() as session:
        yield session


def get_queue(request: Request) -> Queue:
    return request.app.state.queue


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


SessionDep = Annotated[Session, Depends(get_session)]
QueueDep = Annotated[Queue, Depends(get_queue)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _job_or_404(session: Session, document_id: uuid.UUID) -> ProcessingJob:
    job = repository.latest_job(session, document_id)
    if job is None:
        raise ApiError(404, "not_found", "no contract with this id")
    return job


def create_app(
    settings: Settings | None = None,
    lifespan=default_lifespan,
    reports_dir: Path = REPORTS_DIR,
) -> FastAPI:
    """Without ``settings``, the default lifespan loads them from the environment."""
    app = FastAPI(
        title="Retrieval Labs API",
        version="0.1.0",
        lifespan=lifespan,
        responses={422: ERRORS, 500: ERRORS},
    )
    if settings is not None:
        app.state.settings = settings
    install_error_handlers(app)

    @app.middleware("http")
    async def reject_oversized_bodies(request: Request, call_next):
        # Refuse before the multipart parser spools anything; decode_contract re-checks the
        # file itself, which also covers requests without a Content-Length.
        length = request.headers.get("content-length")
        limit = request.app.state.settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        if length is not None and (not length.isdigit() or int(length) > limit):
            return error_response(413, "file_too_large", "the request body is too large")
        return await call_next(request)

    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    def health(request: Request) -> Health | JSONResponse:
        checks = {"database": "ok", "queue": "ok"}
        try:
            with request.app.state.sessions() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            checks["database"] = "unavailable"
        try:
            request.app.state.queue.connection.ping()
        except Exception:
            checks["queue"] = "unavailable"
        healthy = all(v == "ok" for v in checks.values())
        body = Health(status="ok" if healthy else "degraded", **checks)
        return body if healthy else JSONResponse(status_code=503, content=body.model_dump())

    @app.post(
        "/contracts",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=Submission,
        responses={413: ERRORS, 415: ERRORS, 503: ERRORS},
    )
    async def submit_contract(
        session: SessionDep,
        queue: QueueDep,
        settings: SettingsDep,
        file: Annotated[UploadFile, File(description="UTF-8 .txt contract")],
        title: Annotated[str | None, Form()] = None,
    ) -> Submission:
        check_file_type(file.filename, file.content_type)
        limit = settings.max_upload_bytes
        contract_text = decode_contract(await file.read(limit + 1), limit)
        filename = safe_filename(file.filename)
        with session.begin():
            document = repository.create_document(
                session,
                title=clean_title(title, filename),
                filename=filename,
                text=contract_text,
            )
            job = repository.create_job(
                session,
                document.id,
                retriever=settings.retriever,
                llm_provider=settings.llm_provider,
            )
        try:
            enqueue_job(queue, job.id, settings.job_timeout_s)
        except Exception as error:
            logger.error("enqueue failed", extra={"job_id": str(job.id)}, exc_info=error)
            with session.begin():
                repository.claim_job(session, job.id)
                repository.finish_job(session, job.id, JobState.FAILED, "queue_unavailable")
            raise ApiError(503, "queue_unavailable", "the job queue is unavailable") from None
        logger.info(
            "contract submitted",
            extra={
                "document_id": str(document.id),
                "job_id": str(job.id),
                "text_sha256": document.text_sha256,
                "characters": len(contract_text),
            },
        )
        return Submission(
            document_id=document.id,
            job_id=job.id,
            state=JobState.PENDING,
            status_url=f"/contracts/{document.id}/status",
            report_url=f"/contracts/{document.id}/report",
        )

    @app.get("/contracts/{document_id}/status", response_model=JobStatus, responses={404: ERRORS})
    def contract_status(
        document_id: uuid.UUID, session: SessionDep, settings: SettingsDep
    ) -> JobStatus:
        with session.begin():
            repository.fail_lost_jobs(session, settings.lost_after)
        job = _job_or_404(session, document_id)
        return JobStatus(
            document_id=job.document_id,
            job_id=job.id,
            state=job.state,
            error_code=job.error_code,
            retriever=job.retriever,
            llm_provider=job.llm_provider,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    @app.get(
        "/contracts/{document_id}/report",
        response_model=Report,
        responses={404: ERRORS, 409: ERRORS},
    )
    def contract_report(document_id: uuid.UUID, session: SessionDep) -> Report:
        job = _job_or_404(session, document_id)
        if job.state is JobState.FAILED:
            raise ApiError(409, "job_failed", f"the analysis failed ({job.error_code})")
        report = repository.get_report(session, job.id)
        if job.state is not JobState.COMPLETED or report is None:
            raise ApiError(409, "report_not_ready", f"the analysis is {job.state.value}")
        return Report(document_id=document_id, job_id=job.id, report=report)

    @app.delete(
        "/contracts/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={404: ERRORS, 409: ERRORS},
    )
    def delete_contract(document_id: uuid.UUID, session: SessionDep) -> Response:
        with session.begin():
            job = _job_or_404(session, document_id)
            if job.state is JobState.RUNNING:
                raise ApiError(409, "job_running", "the analysis is running; retry later")
            repository.delete_document(session, document_id)
        logger.info("contract deleted", extra={"document_id": str(document_id)})
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/benchmarks/latest", response_model=Benchmark)
    def latest_benchmark() -> Benchmark:
        return Benchmark(
            run=TEST_RUN,
            split="test",
            note=(
                "Single run on the CUAD test split (102 contracts) after all methods were "
                "frozen on dev; brackets are 95% paired bootstrap intervals. bm25-expanded "
                "was tuned on dev."
            ),
            rows=load_rows(TEST_RUN, reports_dir),
        )

    return app


app = create_app()
