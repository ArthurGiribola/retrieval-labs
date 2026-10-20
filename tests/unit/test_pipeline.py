from pathlib import Path

import pytest

from retrieval_labs.clauses import ClauseType, ReportSection
from retrieval_labs.documents import Contract
from retrieval_labs.extraction.budget import BudgetExceededError, SpendLedger
from retrieval_labs.extraction.llm import ExtractionRequest, FakeExtractor, empty_answer
from retrieval_labs.extraction.schema import ClauseAnswer, GoverningLawValue, Quote, Status
from retrieval_labs.ingestion.chunking import ChunkingConfig
from retrieval_labs.pipeline import ContractReport, PipelineConfig, analyze_contract
from retrieval_labs.retrieval.bm25 import BM25Retriever

LAW = "This Agreement shall be governed by the laws of the State of New York."
TEXT = (
    "SUPPLY AGREEMENT\n\n"
    "1. Products. Supplier shall deliver the products listed in Exhibit A.\n"
    f"2. Governing Law. {LAW}\n"
    "3. Payment. Buyer shall pay each invoice within thirty days.\n"
)
# Overlap 14 guarantees any run of up to 15 words (LAW has 13) fits in one chunk. With a
# smaller overlap the quote can straddle chunks, fail verification and become UNCERTAIN.
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


def run(extractor: FakeExtractor, ledger: SpendLedger | None = None) -> ContractReport:
    contract = Contract.from_text("c1", "Supply Agreement", TEXT)
    return analyze_contract(contract, BM25Retriever(), extractor, CONFIG, "test-run", ledger)


def test_end_to_end_with_fake_extractor() -> None:
    report = run(FakeExtractor(responder=quote_law))
    assert [r.clause_type for r in report.results] == list(ClauseType)
    law = report.results[0]
    assert law.status == Status.FOUND
    assert law.extracted_text == LAW
    evidence = law.evidence[0]
    assert TEXT[evidence.source_start : evidence.source_end] == LAW
    assert evidence.chunk_id in law.retrieved_chunk_ids
    assert report.run.retriever == "bm25"
    assert report.run.chunking == "words20-overlap14"
    assert {"chunking", "indexing", "retrieval", "extraction", "total"} <= set(report.timings_s)


def test_clauses_without_candidates_skip_the_model() -> None:
    fake = FakeExtractor()
    report = run(fake)
    skipped = [r for r in report.results if not r.retrieved_chunk_ids]
    assert len(fake.requests) == len(ClauseType) - len(skipped)
    for result in skipped:
        assert result.status == Status.NOT_FOUND
        assert "no model call" in result.extraction_notes[0]


def test_report_groups_renewal_clauses_in_one_section() -> None:
    sections = run(FakeExtractor()).by_section()
    assert list(sections) == list(ReportSection)
    assert [r.clause_type for r in sections[ReportSection.RENEWAL]] == [
        ClauseType.RENEWAL_TERM,
        ClauseType.RENEWAL_NOTICE_PERIOD,
    ]


def test_report_json_round_trip() -> None:
    report = run(FakeExtractor(responder=quote_law))
    assert ContractReport.model_validate_json(report.model_dump_json()) == report


class ExpensiveFake(FakeExtractor):
    def worst_case_cost_usd(self, request: ExtractionRequest) -> float:
        return 1.0


def test_budget_is_checked_before_any_call(tmp_path: Path) -> None:
    fake = ExpensiveFake()
    ledger = SpendLedger(tmp_path / "ledger.jsonl", "r", run_cap_usd=0.5)
    with pytest.raises(BudgetExceededError):
        run(fake, ledger)
    assert fake.requests == []


def test_every_call_is_recorded_in_the_ledger(tmp_path: Path) -> None:
    fake = FakeExtractor()
    path = tmp_path / "ledger.jsonl"
    run(fake, SpendLedger(path, "r", run_cap_usd=1.0))
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line]
    assert len(lines) == len(fake.requests)
