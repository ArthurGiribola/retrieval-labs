"""The function RQ runs for each queued job, and the per-process state it uses.

``worker.main`` builds the runtime (models included) once and installs it with ``configure``;
the worker runs jobs in that same process, so models are never reloaded per job.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.db import repository
from retrieval_labs.extraction.budget import SpendLedger
from retrieval_labs.extraction.llm import Extractor
from retrieval_labs.pipeline import PipelineConfig
from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.embeddings import Embedder
from retrieval_labs.worker.process import DEFAULT_CONFIG, process_job
from retrieval_labs.worker.queue import lost_after

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    sessions: sessionmaker[Session]
    retriever: Retriever
    extractor: Extractor
    embedder: Embedder | None = None
    ledger: SpendLedger | None = None
    config: PipelineConfig = DEFAULT_CONFIG
    lost_after: timedelta = field(default_factory=lost_after)


_runtime: Runtime | None = None


def configure(runtime: Runtime | None) -> None:
    global _runtime
    _runtime = runtime


def sweep_lost_jobs(runtime: Runtime) -> list[uuid.UUID]:
    with runtime.sessions.begin() as session:
        lost = repository.fail_lost_jobs(session, runtime.lost_after)
    for job_id in lost:
        logger.warning("job marked lost", extra={"job_id": str(job_id)})
    return lost


def run_job(job_id: str) -> str:
    if _runtime is None:
        raise RuntimeError("worker runtime is not configured")
    sweep_lost_jobs(_runtime)
    outcome = process_job(
        _runtime.sessions,
        uuid.UUID(job_id),
        retriever=_runtime.retriever,
        extractor=_runtime.extractor,
        config=_runtime.config,
        embedder=_runtime.embedder,
        ledger=_runtime.ledger,
    )
    return outcome.value
