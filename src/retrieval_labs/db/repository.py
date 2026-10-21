"""Reads and writes for the application tables. Callers own the session and its transaction."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import timedelta

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from retrieval_labs.db.models import AnalysisResult, Chunk, Document, JobState, ProcessingJob
from retrieval_labs.documents import Contract, sha256_text
from retrieval_labs.ingestion.chunking import Chunk as TextChunk
from retrieval_labs.pipeline import ContractReport

WORKER_LOST = "worker_lost"


def create_document(session: Session, *, title: str, filename: str, text: str) -> Document:
    document = Document(title=title, filename=filename, text=text, text_sha256=sha256_text(text))
    session.add(document)
    session.flush()
    return document


class CorruptedDocumentError(Exception):
    """The stored text no longer matches the hash recorded when it was submitted."""


def load_contract(session: Session, document_id: uuid.UUID) -> Contract | None:
    document = session.get(Document, document_id)
    if document is None:
        return None
    if sha256_text(document.text) != document.text_sha256:
        raise CorruptedDocumentError(str(document_id))
    return Contract(
        contract_id=str(document.id),
        title=document.title,
        text=document.text,
        text_sha256=document.text_sha256,
    )


def create_job(
    session: Session, document_id: uuid.UUID, *, retriever: str, llm_provider: str
) -> ProcessingJob:
    job = ProcessingJob(document_id=document_id, retriever=retriever, llm_provider=llm_provider)
    session.add(job)
    session.flush()
    return job


def get_job(session: Session, job_id: uuid.UUID) -> ProcessingJob | None:
    return session.get(ProcessingJob, job_id)


def claim_job(session: Session, job_id: uuid.UUID) -> bool:
    """Move a PENDING job to RUNNING. False if it is missing or any other worker, or an
    earlier delivery of the same job, already took it: the update is the lock."""
    result = session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id, ProcessingJob.state == JobState.PENDING)
        .values(state=JobState.RUNNING, started_at=func.now())
    )
    return result.rowcount == 1


def finish_job(
    session: Session, job_id: uuid.UUID, state: JobState, error_code: str | None = None
) -> bool:
    """Move a RUNNING job to a final state. False if it is no longer RUNNING (for example,
    marked lost by ``fail_lost_jobs``)."""
    if state not in (JobState.COMPLETED, JobState.FAILED):
        raise ValueError(f"not a final state: {state}")
    result = session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id, ProcessingJob.state == JobState.RUNNING)
        .values(state=state, error_code=error_code, finished_at=func.now())
    )
    return result.rowcount == 1


def fail_lost_jobs(session: Session, older_than: timedelta) -> list[uuid.UUID]:
    """Mark FAILED (``worker_lost``) every job RUNNING for longer than ``older_than``.

    Callers pass more than the queue's job timeout, so a job still inside its time limit is
    never touched. Lost jobs are not re-queued: they may already have made paid calls (D-003).
    """
    result = session.execute(
        update(ProcessingJob)
        .where(
            ProcessingJob.state == JobState.RUNNING,
            ProcessingJob.started_at < func.now() - older_than,
        )
        .values(state=JobState.FAILED, error_code=WORKER_LOST, finished_at=func.now())
        .returning(ProcessingJob.id)
    )
    return list(result.scalars())


def save_chunks(
    session: Session,
    document_id: uuid.UUID,
    chunks: Sequence[TextChunk],
    embeddings: np.ndarray | None = None,
) -> None:
    """Replace the document's chunks. Only offsets are stored; the text is in the document."""
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(f"{len(embeddings)} embeddings for {len(chunks)} chunks")
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    session.add_all(
        Chunk(
            document_id=document_id,
            chunk_index=chunk.index,
            chunk_id=chunk.chunk_id,
            chunking=chunk.chunking,
            start=chunk.start,
            end=chunk.end,
            embedding=None if embeddings is None else embeddings[i],
        )
        for i, chunk in enumerate(chunks)
    )


def save_report(session: Session, job_id: uuid.UUID, report: ContractReport) -> None:
    session.add(
        AnalysisResult(
            job_id=job_id,
            document_id=uuid.UUID(report.contract_id),
            run_id=report.run.run_id,
            report=report.model_dump(mode="json"),
        )
    )


def get_report(session: Session, job_id: uuid.UUID) -> ContractReport | None:
    stored = session.scalars(select(AnalysisResult).where(AnalysisResult.job_id == job_id)).first()
    return None if stored is None else ContractReport.model_validate(stored.report)


def latest_job(session: Session, document_id: uuid.UUID) -> ProcessingJob | None:
    return session.scalars(
        select(ProcessingJob)
        .where(ProcessingJob.document_id == document_id)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id)
        .limit(1)
    ).first()


def document_exists(session: Session, document_id: uuid.UUID) -> bool:
    return session.get(Document, document_id) is not None


def delete_document(session: Session, document_id: uuid.UUID) -> None:
    """Delete the document; its chunks, jobs and reports go with it (ON DELETE CASCADE)."""
    session.execute(delete(Document).where(Document.id == document_id))
