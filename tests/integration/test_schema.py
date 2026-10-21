"""The migrations build the schema the models describe, and the schema enforces its rules."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session

from retrieval_labs.db.models import (
    EMBEDDING_DIMENSION,
    AnalysisResult,
    Base,
    Chunk,
    Document,
    JobState,
    ProcessingJob,
)

pytestmark = pytest.mark.integration

TABLES = {"documents", "chunks", "processing_jobs", "analysis_results"}


GOVERNING_LAW = "This Agreement is governed by the laws of Delaware."


def make_document(text_value: str = GOVERNING_LAW) -> Document:
    return Document(title="Example", filename="example.txt", text=text_value, text_sha256="0" * 64)


def test_migrated_schema_matches_the_models(engine: Engine) -> None:
    with engine.connect() as connection:
        diffs = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert diffs == []


def test_downgrade_removes_everything_and_upgrade_restores_it(
    engine: Engine, migrations: Config
) -> None:
    command.downgrade(migrations, "base")
    with engine.connect() as connection:
        assert TABLES.isdisjoint(inspect(connection).get_table_names())
        types = connection.execute(text("SELECT typname FROM pg_type WHERE typname = 'job_state'"))
        assert types.all() == []
    command.upgrade(migrations, "head")
    with engine.connect() as connection:
        assert set(inspect(connection).get_table_names()) >= TABLES


def test_round_trip_with_embedding_and_cascade_delete(session: Session) -> None:
    document = make_document()
    session.add(document)
    session.flush()
    vector = np.linspace(-1.0, 1.0, EMBEDDING_DIMENSION, dtype=np.float32)
    session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            chunk_id="doc::0",
            chunking="w150-o50",
            start=0,
            end=len(document.text),
            embedding=vector,
        )
    )
    job = ProcessingJob(document_id=document.id, retriever="bm25", llm_provider="fake")
    session.add(job)
    session.flush()
    session.add(
        AnalysisResult(job_id=job.id, document_id=document.id, run_id="r1", report={"ok": True})
    )
    session.commit()
    session.expire_all()

    stored = session.scalars(select(Chunk)).one()
    np.testing.assert_allclose(stored.embedding, vector, rtol=1e-6)
    assert session.get(ProcessingJob, job.id).state is JobState.PENDING
    assert session.get(AnalysisResult, job.id).report == {"ok": True}
    nearest = session.scalars(
        select(Chunk).order_by(Chunk.embedding.cosine_distance(vector)).limit(1)
    ).one()
    assert nearest.chunk_id == "doc::0"

    session.delete(session.get(Document, document.id))
    session.commit()
    for model in (Chunk, ProcessingJob, AnalysisResult):
        assert session.scalars(select(model)).all() == []


def test_empty_text_is_rejected(session: Session) -> None:
    session.add(make_document(text_value=""))
    with pytest.raises(IntegrityError):
        session.commit()


def test_chunk_span_must_be_valid(session: Session) -> None:
    document = make_document()
    session.add(document)
    session.flush()
    session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            chunk_id="doc::0",
            chunking="w150-o50",
            start=10,
            end=10,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_job_state_outside_the_enum_is_rejected(session: Session) -> None:
    document = make_document()
    session.add(document)
    session.commit()
    with pytest.raises(DataError):
        session.execute(
            text(
                "INSERT INTO processing_jobs (id, document_id, state, retriever, llm_provider)"
                " VALUES (:id, :doc, 'DONE', 'bm25', 'fake')"
            ),
            {"id": uuid.uuid4(), "doc": document.id},
        )


def test_embedding_dimension_is_enforced(session: Session) -> None:
    document = make_document()
    session.add(document)
    session.flush()
    session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            chunk_id="doc::0",
            chunking="w150-o50",
            start=0,
            end=5,
            embedding=np.zeros(EMBEDDING_DIMENSION - 1, dtype=np.float32),
        )
    )
    with pytest.raises(DataError):
        session.commit()
