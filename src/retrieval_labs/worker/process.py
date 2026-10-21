"""Process one job: load the stored contract, run the measured pipeline, persist the results.

The retriever and extractor are built once per worker process and passed in, so this module
never loads models itself. Jobs are never retried (D-003): a failure ends in FAILED, and
running the same job again (a duplicate delivery) does nothing, so no paid call is repeated.
"""

from __future__ import annotations

import logging
import uuid
from enum import StrEnum

from rq.timeouts import JobTimeoutException
from sqlalchemy.orm import Session, sessionmaker

from retrieval_labs.db import repository
from retrieval_labs.db.models import JobState
from retrieval_labs.extraction.budget import BudgetExceededError, SpendLedger
from retrieval_labs.extraction.llm import Extractor
from retrieval_labs.ingestion.chunking import chunk_contract
from retrieval_labs.pipeline import PipelineConfig, analyze_contract
from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = PipelineConfig()  # the measured setup: 150/50 chunks, top 5


class Outcome(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"  # missing, or not PENDING (already taken by another delivery)
    DISCARDED = "DISCARDED"  # marked worker_lost while running; its results are not saved


class ErrorCode(StrEnum):
    DOCUMENT_MISSING = "document_missing"
    DOCUMENT_CORRUPTED = "document_corrupted"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    PROCESSING_ERROR = "processing_error"


def process_job(
    sessions: sessionmaker[Session],
    job_id: uuid.UUID,
    *,
    retriever: Retriever,
    extractor: Extractor,
    config: PipelineConfig = DEFAULT_CONFIG,
    embedder: Embedder | None = None,
    ledger: SpendLedger | None = None,
) -> Outcome:
    """With ``embedder`` (dense methods), chunk embeddings are stored too; a caching embedder
    returns the vectors the retriever just computed."""
    with sessions.begin() as session:
        if not repository.claim_job(session, job_id):
            logger.info("job skipped", extra={"job_id": str(job_id)})
            return Outcome.SKIPPED

    try:
        with sessions() as session:
            job = repository.get_job(session, job_id)
            contract = repository.load_contract(session, job.document_id) if job else None
        if contract is None:
            return _fail(sessions, job_id, ErrorCode.DOCUMENT_MISSING)

        report = analyze_contract(contract, retriever, extractor, config, str(job_id), ledger)
        chunks = chunk_contract(contract, config.chunking)
        embeddings = embedder.embed_documents([c.text for c in chunks]) if embedder else None

        with sessions.begin() as session:
            repository.save_chunks(session, uuid.UUID(contract.contract_id), chunks, embeddings)
            repository.save_report(session, job_id, report)
            if not repository.finish_job(session, job_id, JobState.COMPLETED):
                raise _JobLostError  # rolls back the chunks and the report
    except _JobLostError:
        logger.warning("job no longer running; results discarded", extra={"job_id": str(job_id)})
        return Outcome.DISCARDED
    except repository.CorruptedDocumentError as error:
        return _fail(sessions, job_id, ErrorCode.DOCUMENT_CORRUPTED, error)
    except BudgetExceededError as error:
        return _fail(sessions, job_id, ErrorCode.BUDGET_EXCEEDED, error)
    except JobTimeoutException as error:
        return _fail(sessions, job_id, ErrorCode.TIMEOUT, error)
    except Exception as error:
        return _fail(sessions, job_id, ErrorCode.PROCESSING_ERROR, error)

    logger.info(
        "job completed",
        extra={"job_id": str(job_id), "chunks": len(chunks), "timings_s": report.timings_s},
    )
    return Outcome.COMPLETED


class _JobLostError(Exception):
    pass


def _fail(
    sessions: sessionmaker[Session],
    job_id: uuid.UUID,
    code: ErrorCode,
    error: BaseException | None = None,
) -> Outcome:
    # The traceback goes to the worker log only; the database keeps just the short code.
    logger.error(
        "job failed",
        extra={"job_id": str(job_id), "error_code": code.value},
        exc_info=error,
    )
    with sessions.begin() as session:
        repository.finish_job(session, job_id, JobState.FAILED, code.value)
    return Outcome.FAILED
