import pytest
from pydantic import ValidationError

from retrieval_labs.clauses import ClauseType, ReportSection
from retrieval_labs.extraction.evidence import (
    Passage,
    build_result,
    locate_quote,
    select_passages,
)
from retrieval_labs.extraction.schema import (
    ClauseResult,
    Evidence,
    GoverningLawAnswer,
    GoverningLawValue,
    Quote,
    RenewalNoticeValue,
    Status,
)
from retrieval_labs.retrieval.base import RetrievalResult

TEXT = (
    "1. Term. The term is one year.\n"
    "2. Law. This Agreement shall be governed by the laws of the State of New York.\n"
    "3. Notice. Notices must be in writing and “delivered” by hand.\n"
)
LAW = "This Agreement shall be governed by the laws of the State of New York."
NY = GoverningLawValue(jurisdiction="New York")


def result_for(fragment: str, rank: int, chunk: str) -> RetrievalResult:
    start = TEXT.index(fragment)
    return RetrievalResult(
        chunk_id=chunk,
        contract_id="c1",
        rank=rank,
        score=1.0,
        start=start,
        end=start + len(fragment),
        text=fragment,
    )


def whole_text_passage() -> Passage:
    return Passage("P1", 0, len(TEXT), TEXT, (("c1#0000", 0, len(TEXT)),))


def answer(status: Status, *quotes: tuple[str, str], value=NY) -> GoverningLawAnswer:
    return GoverningLawAnswer(
        status=status,
        quotes=[Quote(passage_id=p, text=t) for p, t in quotes],
        notes="",
        value=value,
    )


def build(ans: GoverningLawAnswer | None, passages=None) -> ClauseResult:
    return build_result(
        ClauseType.GOVERNING_LAW,
        ans,
        passages or [whole_text_passage()],
        "c1",
        TEXT,
        ["c1#0000"],
        failure_note="model declined",
    )


# --- passage selection ---


def test_passages_merge_overlaps_and_follow_document_order() -> None:
    results = [
        result_for("3. Notice.", 1, "c1#0002"),
        result_for("1. Term. The term", 2, "c1#0000"),
        result_for("The term is one year.", 3, "c1#0001"),  # overlaps the previous one
    ]
    passages = select_passages(results, TEXT)
    assert [p.passage_id for p in passages] == ["P1", "P2"]
    assert passages[0].text == "1. Term. The term is one year."
    assert [c[0] for c in passages[0].chunks] == ["c1#0000", "c1#0001"]
    assert passages[1].text == "3. Notice."
    assert all(TEXT[p.start : p.end] == p.text for p in passages)


def test_no_results_means_no_passages() -> None:
    assert select_passages([], TEXT) == []


# --- quote location ---


def test_locate_exact_quote() -> None:
    start, end = locate_quote(LAW, whole_text_passage())
    assert TEXT[start:end] == LAW


def test_locate_quote_with_different_whitespace() -> None:
    passage = whole_text_passage()
    start, end = locate_quote("This   Agreement shall be\ngoverned by the laws", passage)
    assert TEXT[start:end] == "This Agreement shall be governed by the laws"


def test_locate_quote_with_straight_instead_of_curly_quotes() -> None:
    start, end = locate_quote('and "delivered" by hand', whole_text_passage())
    assert TEXT[start:end] == "and “delivered” by hand"


def test_fabricated_or_paraphrased_quotes_are_not_located() -> None:
    passage = whole_text_passage()
    assert locate_quote("governed by the laws of Delaware", passage) is None
    assert locate_quote("This Agreement is governed by New York law.", passage) is None
    assert locate_quote("   ", passage) is None


def test_locate_uses_absolute_offsets() -> None:
    start = TEXT.index("2. Law.")
    passage = Passage("P1", start, len(TEXT), TEXT[start:], (("c1#0001", start, len(TEXT)),))
    s, e = locate_quote("State of New York", passage)
    assert TEXT[s:e] == "State of New York"


# --- result building ---


def test_found_with_verified_quote() -> None:
    result = build(answer(Status.FOUND, ("P1", LAW)))
    assert result.status == Status.FOUND
    assert result.extracted_text == LAW
    assert result.evidence[0].source_start == TEXT.index(LAW)
    assert result.normalized_value == NY
    assert result.section == ReportSection.GOVERNING_LAW


def test_extracted_text_is_the_contract_slice_not_the_model_string() -> None:
    result = build(answer(Status.FOUND, ("P1", "governed  by the\nlaws")))
    assert result.extracted_text == "governed by the laws"


def test_quote_cited_with_wrong_passage_id_is_still_found() -> None:
    passages = [
        Passage("P1", 0, 30, TEXT[:30], (("c1#0000", 0, 30),)),
        Passage("P2", 31, len(TEXT), TEXT[31:], (("c1#0001", 31, len(TEXT)),)),
    ]
    result = build(answer(Status.FOUND, ("P1", LAW)), passages)
    assert result.status == Status.FOUND
    assert result.evidence[0].chunk_id == "c1#0001"


def test_found_without_verifiable_quote_is_downgraded() -> None:
    result = build(answer(Status.FOUND, ("P1", "governed by the laws of Delaware")))
    assert result.status == Status.UNCERTAIN
    assert result.evidence == []
    assert result.extracted_text is None
    assert any("rejected" in n for n in result.extraction_notes)
    assert any("downgraded" in n for n in result.extraction_notes)


def test_partially_verified_quotes_keep_only_verified_evidence() -> None:
    result = build(answer(Status.FOUND, ("P1", "invented text"), ("P1", LAW)))
    assert result.status == Status.FOUND
    assert [e.exact_quote for e in result.evidence] == [LAW]
    assert any("1 quote(s) rejected" in n for n in result.extraction_notes)


def test_evidence_is_deduplicated_and_keeps_the_models_relevance_order() -> None:
    notice = "Notices must be in writing"  # appears after LAW in the contract
    result = build(answer(Status.FOUND, ("P1", notice), ("P1", LAW), ("P1", notice)))
    assert [e.exact_quote for e in result.evidence] == [notice, LAW]
    assert result.extracted_text == notice


def test_not_found_drops_quotes_and_value() -> None:
    result = build(answer(Status.NOT_FOUND, ("P1", LAW)))
    assert result.status == Status.NOT_FOUND
    assert (result.evidence, result.extracted_text, result.normalized_value) == ([], None, None)
    assert any("ignored" in n for n in result.extraction_notes)


def test_uncertain_keeps_verified_candidates() -> None:
    result = build(answer(Status.UNCERTAIN, ("P1", LAW)))
    assert result.status == Status.UNCERTAIN
    assert result.extracted_text == LAW


def test_missing_answer_is_uncertain_with_reason() -> None:
    result = build(None)
    assert result.status == Status.UNCERTAIN
    assert result.extraction_notes == ["model declined"]


# --- result schema invariants ---


def evidence() -> Evidence:
    start = TEXT.index(LAW)
    return Evidence(
        document_id="c1",
        chunk_id="c1#0000",
        source_start=start,
        source_end=start + len(LAW),
        exact_quote=LAW,
    )


def make_result(**overrides) -> dict:
    data = {
        "clause_type": ClauseType.GOVERNING_LAW,
        "status": Status.FOUND,
        "extracted_text": LAW,
        "normalized_value": NY,
        "evidence": [evidence()],
        "extraction_notes": [],
        "retrieved_chunk_ids": [],
    }
    return {**data, **overrides}


def test_result_rejects_found_without_evidence() -> None:
    with pytest.raises(ValidationError, match="FOUND requires"):
        ClauseResult(**make_result(evidence=[], extracted_text=None))


def test_result_rejects_extracted_text_not_from_evidence() -> None:
    with pytest.raises(ValidationError, match="first verified evidence"):
        ClauseResult(**make_result(extracted_text="a paraphrase"))


def test_result_rejects_value_of_another_clause_type() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        ClauseResult(**make_result(normalized_value=RenewalNoticeValue(notice_days=30)))


def test_result_json_round_trip_keeps_value_type() -> None:
    result = ClauseResult(**make_result())
    again = ClauseResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert type(again.normalized_value) is GoverningLawValue
