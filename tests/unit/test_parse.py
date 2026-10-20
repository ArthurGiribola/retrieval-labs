from pathlib import Path
from typing import Any

import pytest

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.parse import make_contract_id, parse_archive
from synthetic_cuad import (
    ALPHA,
    ALPHA_TEXT,
    BETA,
    SYNTHETIC,
    SYNTHETIC_TEST,
    SYNTHETIC_TRAIN,
    ArchiveFactory,
)


def test_contract_id_is_stable_and_safe() -> None:
    contract_id = make_contract_id(BETA)
    assert contract_id == make_contract_id(BETA)
    assert contract_id.startswith("cuad_")
    assert len(contract_id) == len("cuad_") + 12
    assert contract_id.replace("_", "").isalnum()


def test_parses_all_contracts_with_exact_text(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    assert parsed.json_version == "aok_v1.0"
    assert {c.title for c in parsed.contracts} == set(SYNTHETIC)
    alpha = next(c for c in parsed.contracts if c.title == ALPHA)
    assert alpha.text == ALPHA_TEXT  # leading whitespace preserved
    assert [c.contract_id for c in parsed.contracts] == sorted(
        c.contract_id for c in parsed.contracts
    )


def test_one_annotation_per_contract_and_target_label(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    assert len(parsed.gold) == len(SYNTHETIC) * len(CuadLabel)
    assert {g.label for g in parsed.gold} == set(CuadLabel)  # "Parties" ignored


def test_absent_labels_are_kept_as_empty_annotations(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    beta = [g for g in parsed.gold if g.contract_id == make_contract_id(BETA)]
    assert len(beta) == len(CuadLabel)
    assert not any(g.is_answerable for g in beta)


def test_multiple_spans_are_sorted_and_match_text(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    cap = next(
        g
        for g in parsed.gold
        if g.contract_id == make_contract_id(ALPHA) and g.label == CuadLabel.CAP_ON_LIABILITY
    )
    assert [s.text for s in cap.spans] == [
        "Liability is capped at fees paid.",
        "Neither party is liable for indirect damages.",
    ]
    assert all(ALPHA_TEXT[s.start : s.end] == s.text for s in cap.spans)


def test_official_split_membership(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    assert parsed.official_train_ids == {make_contract_id(t) for t in SYNTHETIC_TRAIN}
    assert parsed.official_test_ids == {make_contract_id(t) for t in SYNTHETIC_TEST}


def _first_target_answer(docs: list[dict[str, Any]]) -> dict[str, Any]:
    qas = docs[0]["paragraphs"][0]["qas"]
    return next(qa for qa in qas if qa["answers"] and "Governing Law" in qa["id"])


def test_rejects_offset_mismatch(make_cuad_archive: ArchiveFactory) -> None:
    def shift(docs: list[dict[str, Any]]) -> None:
        _first_target_answer(docs)["answers"][0]["answer_start"] += 1

    with pytest.raises(CuadDataError, match="offset mismatch"):
        parse_archive(make_cuad_archive(mutate=shift))


def test_rejects_inconsistent_is_impossible(make_cuad_archive: ArchiveFactory) -> None:
    def flip(docs: list[dict[str, Any]]) -> None:
        _first_target_answer(docs)["is_impossible"] = True

    with pytest.raises(CuadDataError, match="is_impossible"):
        parse_archive(make_cuad_archive(mutate=flip))


def test_rejects_label_mismatch_between_id_and_question(
    make_cuad_archive: ArchiveFactory,
) -> None:
    def relabel(docs: list[dict[str, Any]]) -> None:
        qa = _first_target_answer(docs)
        qa["question"] = qa["question"].replace("Governing Law", "Renewal Term")

    with pytest.raises(CuadDataError, match="label differs"):
        parse_archive(make_cuad_archive(mutate=relabel))


def test_rejects_missing_target_question(make_cuad_archive: ArchiveFactory) -> None:
    def drop(docs: list[dict[str, Any]]) -> None:
        paragraph = docs[0]["paragraphs"][0]
        paragraph["qas"] = [qa for qa in paragraph["qas"] if "Renewal Term" not in qa["id"]]

    with pytest.raises(CuadDataError, match="missing questions"):
        parse_archive(make_cuad_archive(mutate=drop))


def test_rejects_title_in_both_splits(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive(test=[*SYNTHETIC_TEST, ALPHA])
    with pytest.raises(CuadDataError, match="both train and test"):
        parse_archive(archive)


def test_rejects_title_in_no_split(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive(train=SYNTHETIC_TRAIN[1:])
    with pytest.raises(CuadDataError, match="in no split"):
        parse_archive(archive)


def test_rejects_duplicate_title(make_cuad_archive: ArchiveFactory) -> None:
    def duplicate(docs: list[dict[str, Any]]) -> None:
        docs.append(docs[0])

    with pytest.raises(CuadDataError, match="duplicate title"):
        parse_archive(make_cuad_archive(mutate=duplicate))
