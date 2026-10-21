"""Tables for submitted contracts, their chunks, processing jobs and analysis results.

Changes here need a matching migration in ``migrations/versions/``; a test compares the
migrated schema with this metadata. Alerts (Phase 5) and benchmark runs are not stored yet.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# bge-small-en-v1.5 output size (retrieval/embeddings.py, BGE_SMALL).
EMBEDDING_DIMENSION = 384

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Document(Base):
    """A submitted contract. ``text`` is stored exactly as received (offsets refer to it)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (CheckConstraint("length(text) > 0", name="text_not_empty"),)


class Chunk(Base):
    """A passage of a document, by character offsets into ``documents.text``."""

    __tablename__ = "chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(255))
    chunking: Mapped[str] = mapped_column(String(64))
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    # Null when the job's retriever does not embed (BM25).
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)

    __table_args__ = (CheckConstraint('start >= 0 AND start < "end"', name="valid_span"),)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[JobState] = mapped_column(
        Enum(JobState, name="job_state"), default=JobState.PENDING
    )
    retriever: Mapped[str] = mapped_column(String(64))
    llm_provider: Mapped[str] = mapped_column(String(64))
    # Short machine-readable reason for FAILED; details go to the logs, never here.
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AnalysisResult(Base):
    """The pipeline's ``ContractReport`` for one completed job, as JSON."""

    __tablename__ = "analysis_results"

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(String(255))
    report: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()
