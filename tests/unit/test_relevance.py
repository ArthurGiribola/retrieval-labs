import pytest

from retrieval_labs.clauses import ClauseType
from retrieval_labs.cuad.gold import GoldSpan
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.evaluation.labels import CLAUSE_TYPE_OF, CUAD_LABEL_OF
from retrieval_labs.evaluation.relevance import Criterion, is_relevant, overlap, relevant_spans


def span(start: int, end: int) -> GoldSpan:
    return GoldSpan(start=start, end=end, text="x" * (end - start))


def test_label_mapping_is_a_bijection() -> None:
    assert set(CUAD_LABEL_OF) == set(ClauseType)
    assert set(CLAUSE_TYPE_OF) == set(CuadLabel)


def test_overlap() -> None:
    assert overlap(0, 100, span(50, 150)) == 50
    assert overlap(0, 100, span(100, 150)) == 0  # end is exclusive
    assert overlap(0, 100, span(20, 30)) == 10


@pytest.mark.parametrize(
    ("chunk", "gold", "expected"),
    [
        ((0, 1000), (100, 200), True),  # short span fully inside
        ((0, 1000), (950, 1050), True),  # half of a short span inside: 50/100
        ((0, 1000), (960, 1060), False),  # 40% of a short span inside
        ((0, 100), (0, 1000), True),  # span longer than chunk, fills the whole chunk
        ((0, 100), (50, 1000), True),  # fills half the chunk: 50/100
        ((0, 100), (60, 1000), False),  # fills 40% of the chunk
        ((0, 100), (200, 300), False),  # disjoint
    ],
)
def test_primary_criterion(chunk, gold, expected) -> None:
    assert is_relevant(*chunk, span(*gold)) is expected


def test_sensitivity_criteria() -> None:
    straddling = span(950, 1050)
    assert is_relevant(0, 1000, straddling, Criterion.ANY_OVERLAP)
    assert not is_relevant(0, 1000, straddling, Criterion.CONTAINMENT)
    assert is_relevant(0, 1000, span(10, 20), Criterion.CONTAINMENT)
    assert not is_relevant(0, 1000, span(1000, 1010), Criterion.ANY_OVERLAP)


def test_lenient_criterion_accepts_what_primary_rejects() -> None:
    sliver = span(990, 1100)  # 10 of 110 characters inside the chunk
    assert is_relevant(0, 1000, sliver, Criterion.ANY_OVERLAP)
    assert not is_relevant(0, 1000, sliver)


def test_relevant_spans_with_multiple_gold_spans() -> None:
    spans = [span(10, 20), span(500, 600), span(2000, 2100)]
    assert relevant_spans(0, 1000, spans) == [0, 1]
