import pytest
from pydantic import ValidationError

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.gold import GoldAnnotation, GoldSpan, check_offsets
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.documents import Contract, sha256_text

TEXT = "This Agreement shall be governed by the laws of the State of New York."


def make_contract() -> Contract:
    return Contract.from_text("c1", "Sample Agreement", TEXT)


def span_of(fragment: str) -> GoldSpan:
    start = TEXT.index(fragment)
    return GoldSpan(start=start, end=start + len(fragment), text=fragment)


def test_contract_from_text_computes_hash() -> None:
    contract = make_contract()
    assert contract.text_sha256 == sha256_text(TEXT)


def test_contract_rejects_wrong_hash() -> None:
    with pytest.raises(ValidationError, match="text_sha256"):
        Contract(contract_id="c1", title="t", text=TEXT, text_sha256="0" * 64)


def test_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Contract(contract_id="c1", title="t", text=TEXT, text_sha256=sha256_text(TEXT), answers=[])


def test_contract_is_immutable() -> None:
    contract = make_contract()
    with pytest.raises(ValidationError):
        contract.text = "changed"


def test_cuad_label_values_match_dataset_spelling() -> None:
    assert [label.value for label in CuadLabel] == [
        "Governing Law",
        "Termination For Convenience",
        "Cap On Liability",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
    ]


def test_gold_span_requires_consistent_end() -> None:
    with pytest.raises(ValidationError, match="span end"):
        GoldSpan(start=0, end=5, text="abc")


def test_gold_annotation_rejects_unsorted_spans() -> None:
    first, second = span_of("governed"), span_of("New York")
    with pytest.raises(ValidationError, match="sorted"):
        GoldAnnotation(contract_id="c1", label=CuadLabel.GOVERNING_LAW, spans=(second, first))


def test_gold_annotation_rejects_duplicate_spans() -> None:
    span = span_of("New York")
    with pytest.raises(ValidationError, match="unique"):
        GoldAnnotation(contract_id="c1", label=CuadLabel.GOVERNING_LAW, spans=(span, span))


def test_gold_annotation_allows_overlapping_spans() -> None:
    outer, inner = span_of("the State of New York"), span_of("New York")
    annotation = GoldAnnotation(
        contract_id="c1", label=CuadLabel.GOVERNING_LAW, spans=(outer, inner)
    )
    check_offsets(annotation, make_contract())


def test_empty_annotation_is_not_answerable() -> None:
    annotation = GoldAnnotation(contract_id="c1", label=CuadLabel.RENEWAL_TERM, spans=())
    assert not annotation.is_answerable


def test_check_offsets_accepts_matching_spans() -> None:
    annotation = GoldAnnotation(
        contract_id="c1", label=CuadLabel.GOVERNING_LAW, spans=(span_of("New York"),)
    )
    assert annotation.is_answerable
    check_offsets(annotation, make_contract())


def test_check_offsets_detects_shifted_span() -> None:
    real = span_of("New York")
    shifted = GoldSpan(start=real.start + 1, end=real.end + 1, text=real.text)
    annotation = GoldAnnotation(contract_id="c1", label=CuadLabel.GOVERNING_LAW, spans=(shifted,))
    with pytest.raises(CuadDataError, match="offset mismatch"):
        check_offsets(annotation, make_contract())


def test_check_offsets_rejects_other_contract() -> None:
    annotation = GoldAnnotation(contract_id="other", label=CuadLabel.GOVERNING_LAW, spans=())
    with pytest.raises(CuadDataError):
        check_offsets(annotation, make_contract())
