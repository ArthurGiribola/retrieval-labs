"""Synchronous job processing against the real database, with free stand-in models."""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from fake_models import HashingEmbedder
from retrieval_labs.clauses import ClauseType
from retrieval_labs.db import repository
from retrieval_labs.db.models import (
    EMBEDDING_DIMENSION,
    AnalysisResult,
    Chunk,
    Document,
    JobState,
    ProcessingJob,
)
from retrieval_labs.extraction.budget import SpendLedger
from retrieval_labs.extraction.llm import ExtractionRequest, FakeExtractor, empty_answer
from retrieval_labs.extraction.schema import ClauseAnswer, GoverningLawValue, Quote, Status
from retrieval_labs.ingestion.chunking import ChunkingConfig, chunk_contract
from retrieval_labs.pipeline import PipelineConfig, analyze_contract
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.dense import DenseRetriever
from retrieval_labs.worker.process import ErrorCode, Outcome, process_job

pytestmark = pytest.mark.integration

LAW = "This Agreement shall be governed by the laws of the State of New York."
TEXT = (
    "SUPPLY AGREEMENT\n\n"
    "1. Products. Supplier shall deliver the products listed in Exhibit A.\n"
    f"2. Governing Law. {LAW}\n"
    "3. Payment. Buyer shall pay each invoice within thirty days.\n"
)
# Same chunking as tests/unit/test_pipeline.py: the 13-word quote fits in one chunk.
CONFIG = PipelineConfig(chunking=ChunkingConfig(window_words=20, overlap_words=14), top_k=2)


def quote_law(request: ExtractionRequest) -> ClauseAnswer | None:
    if request.clause_type != ClauseType.GOVERNING_LAW:
        return empty_answer(request.clause_type)
    return request.answer_model.model_validate(
        {
            "status": Status.FOUND,
            "quotes": [Quote(passage_id="P1", text=LAW)],
            "notes": "",
            "value": GoverningLawValue(jurisdiction="New York"),
        }
    )


class ExplodingExtractor(FakeExtractor):
    def extract(self, request: ExtractionRequest):
        raise RuntimeError("upstream said: secret-token-123")


class ExpensiveFake(FakeExtractor):
    def worst_case_cost_usd(self, request: ExtractionRequest) -> float:
        return 1.0


def submit(sessions: sessionmaker[Session], retriever: str = "bm25") -> uuid.UUID:
    with sessions.begin() as session:
        document = repository.create_document(
            session, title="Supply Agreement", filename="supply.txt", text=TEXT
        )
        job = repository.create_job(session, document.id, retriever=retriever, llm_provider="fake")
        return job.id


def test_job_completes_and_the_stored_report_matches_the_in_memory_one(
    sessions: sessionmaker[Session],
) -> None:
    job_id = submit(sessions)
    outcome = process_job(
        sessions,
        job_id,
        retriever=BM25Retriever(),
        extractor=FakeExtractor(responder=quote_law),
        config=CONFIG,
    )
    assert outcome is Outcome.COMPLETED

    with sessions() as session:
        job = repository.get_job(session, job_id)
        stored = repository.get_report(session, job_id)
        contract = repository.load_contract(session, job.document_id)
        chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert job.state is JobState.COMPLETED
    assert job.error_code is None
    assert job.started_at is not None and job.finished_at >= job.started_at

    expected = analyze_contract(
        contract, BM25Retriever(), FakeExtractor(responder=quote_law), CONFIG, str(job_id)
    )
    assert stored.model_dump(exclude={"timings_s"}) == expected.model_dump(exclude={"timings_s"})
    law = next(r for r in stored.results if r.clause_type is ClauseType.GOVERNING_LAW)
    assert law.status is Status.FOUND
    assert law.extracted_text == LAW

    expected_chunks = chunk_contract(contract, CONFIG.chunking)
    assert [(c.chunk_id, c.start, c.end) for c in chunks] == [
        (c.chunk_id, c.start, c.end) for c in expected_chunks
    ]
    assert all(
        TEXT[c.start : c.end] == e.text for c, e in zip(chunks, expected_chunks, strict=True)
    )
    assert all(c.embedding is None for c in chunks)


def test_dense_jobs_store_the_chunk_embeddings(sessions: sessionmaker[Session]) -> None:
    job_id = submit(sessions, retriever="dense")
    embedder = HashingEmbedder(dimension=EMBEDDING_DIMENSION)
    outcome = process_job(
        sessions,
        job_id,
        retriever=DenseRetriever(embedder),
        extractor=FakeExtractor(),
        config=CONFIG,
        embedder=embedder,
    )
    assert outcome is Outcome.COMPLETED
    with sessions() as session:
        chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    expected = embedder.embed_documents([TEXT[c.start : c.end] for c in chunks])
    np.testing.assert_allclose(np.stack([c.embedding for c in chunks]), expected, rtol=1e-6)


def test_a_second_delivery_of_the_same_job_does_nothing(sessions: sessionmaker[Session]) -> None:
    job_id = submit(sessions)
    extractor = FakeExtractor()
    assert process_job(sessions, job_id, retriever=BM25Retriever(), extractor=extractor) is (
        Outcome.COMPLETED
    )
    calls = len(extractor.requests)
    assert process_job(sessions, job_id, retriever=BM25Retriever(), extractor=extractor) is (
        Outcome.SKIPPED
    )
    assert len(extractor.requests) == calls
    with sessions() as session:
        assert len(session.scalars(select(AnalysisResult)).all()) == 1


def test_unknown_job_is_skipped(sessions: sessionmaker[Session]) -> None:
    outcome = process_job(
        sessions, uuid.uuid4(), retriever=BM25Retriever(), extractor=FakeExtractor()
    )
    assert outcome is Outcome.SKIPPED


def test_a_failure_is_stored_as_a_short_code_and_logged_with_its_traceback(
    sessions: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    job_id = submit(sessions)
    with caplog.at_level(logging.ERROR, logger="retrieval_labs.worker.process"):
        outcome = process_job(
            sessions, job_id, retriever=BM25Retriever(), extractor=ExplodingExtractor()
        )
    assert outcome is Outcome.FAILED
    with sessions() as session:
        job = repository.get_job(session, job_id)
        assert repository.get_report(session, job_id) is None
        assert session.scalars(select(Chunk)).all() == []
    assert job.state is JobState.FAILED
    assert job.error_code == ErrorCode.PROCESSING_ERROR
    assert job.finished_at is not None
    [record] = caplog.records
    assert record.exc_info is not None
    # The contract text is never logged, even on failure.
    assert all(LAW not in r.getMessage() and LAW not in str(r.__dict__) for r in caplog.records)


def test_budget_exhaustion_fails_the_job_before_any_call(
    sessions: sessionmaker[Session], tmp_path: Path
) -> None:
    job_id = submit(sessions)
    extractor = ExpensiveFake()
    outcome = process_job(
        sessions,
        job_id,
        retriever=BM25Retriever(),
        extractor=extractor,
        ledger=SpendLedger(tmp_path / "ledger.jsonl", "r", run_cap_usd=0.5),
    )
    assert outcome is Outcome.FAILED
    assert extractor.requests == []
    with sessions() as session:
        assert repository.get_job(session, job_id).error_code == ErrorCode.BUDGET_EXCEEDED


def test_a_document_altered_after_submission_is_not_processed(
    sessions: sessionmaker[Session],
) -> None:
    job_id = submit(sessions)
    with sessions.begin() as session:
        session.execute(update(Document).values(text=TEXT.replace("New York", "Delaware")))
    extractor = FakeExtractor()
    outcome = process_job(sessions, job_id, retriever=BM25Retriever(), extractor=extractor)
    assert outcome is Outcome.FAILED
    assert extractor.requests == []
    with sessions() as session:
        assert repository.get_job(session, job_id).error_code == ErrorCode.DOCUMENT_CORRUPTED


def test_results_are_discarded_if_the_job_is_marked_lost_while_running(
    sessions: sessionmaker[Session],
) -> None:
    job_id = submit(sessions)

    def mark_lost_midway(request: ExtractionRequest) -> ClauseAnswer | None:
        with sessions.begin() as session:
            repository.fail_lost_jobs(session, timedelta(0))
        return empty_answer(request.clause_type)

    outcome = process_job(
        sessions,
        job_id,
        retriever=BM25Retriever(),
        extractor=FakeExtractor(responder=mark_lost_midway),
        config=CONFIG,
    )
    assert outcome is Outcome.DISCARDED
    with sessions() as session:
        job = repository.get_job(session, job_id)
        assert repository.get_report(session, job_id) is None
        assert session.scalars(select(Chunk)).all() == []
    assert job.state is JobState.FAILED
    assert job.error_code == repository.WORKER_LOST


def test_only_running_jobs_past_the_limit_are_marked_lost(
    sessions: sessionmaker[Session],
) -> None:
    pending = submit(sessions)
    old = submit(sessions)
    with sessions.begin() as session:
        session.execute(
            update(ProcessingJob)
            .where(ProcessingJob.id == old)
            .values(state=JobState.RUNNING, started_at=func.now() - timedelta(hours=1))
        )
        assert repository.fail_lost_jobs(session, timedelta(minutes=30)) == [old]
        assert repository.fail_lost_jobs(session, timedelta(minutes=30)) == []
    with sessions() as session:
        assert repository.get_job(session, pending).state is JobState.PENDING
