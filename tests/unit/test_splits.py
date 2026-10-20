import pytest

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.splits import Split, assign_splits, duplicate_text_groups
from retrieval_labs.documents import Contract


def make_contracts(n: int, duplicates: dict[int, int] | None = None) -> list[Contract]:
    """``n`` contracts with distinct texts; ``duplicates={i: j}`` gives contract i the text of j."""
    duplicates = duplicates or {}
    return [
        Contract.from_text(f"c{i:03d}", f"title {i}", f"text {duplicates.get(i, i)}")
        for i in range(n)
    ]


CONTRACTS = make_contracts(40)
TRAIN = {c.contract_id for c in CONTRACTS[:30]}
TEST = {c.contract_id for c in CONTRACTS[30:]}


def by_split(assignment: dict[str, Split], split: Split) -> set[str]:
    return {cid for cid, s in assignment.items() if s == split}


def test_sizes_and_official_test_preserved() -> None:
    assignment = assign_splits(CONTRACTS, TRAIN, TEST, dev_size=8, seed=42)
    assert len(assignment) == 40
    assert by_split(assignment, Split.TEST) == TEST
    assert len(by_split(assignment, Split.DEV)) == 8
    assert by_split(assignment, Split.DEV) | by_split(assignment, Split.TRAIN) == TRAIN


def test_dev_comes_only_from_official_train() -> None:
    assignment = assign_splits(CONTRACTS, TRAIN, TEST, dev_size=8, seed=42)
    assert by_split(assignment, Split.DEV) <= TRAIN


def test_deterministic_and_independent_of_input_order() -> None:
    first = assign_splits(CONTRACTS, TRAIN, TEST, dev_size=8, seed=42)
    again = assign_splits(list(reversed(CONTRACTS)), TRAIN, TEST, dev_size=8, seed=42)
    assert first == again


def test_seed_changes_dev_selection() -> None:
    a = assign_splits(CONTRACTS, TRAIN, TEST, dev_size=8, seed=42)
    b = assign_splits(CONTRACTS, TRAIN, TEST, dev_size=8, seed=7)
    assert by_split(a, Split.DEV) != by_split(b, Split.DEV)


def test_identical_texts_always_share_a_split() -> None:
    contracts = make_contracts(40, duplicates={1: 0, 2: 0, 4: 3})
    for seed in range(50):
        assignment = assign_splits(contracts, TRAIN, TEST, dev_size=8, seed=seed)
        assert assignment["c000"] == assignment["c001"] == assignment["c002"]
        assert assignment["c003"] == assignment["c004"]
        assert len(by_split(assignment, Split.DEV)) == 8


def test_identical_text_across_official_splits_is_rejected() -> None:
    contracts = make_contracts(40, duplicates={35: 0})  # c035 (test) duplicates c000 (train)
    with pytest.raises(CuadDataError, match="across official splits"):
        assign_splits(contracts, TRAIN, TEST, dev_size=8)


def test_dev_size_larger_than_train_is_rejected() -> None:
    with pytest.raises(CuadDataError, match="cannot draw"):
        assign_splits(CONTRACTS, TRAIN, TEST, dev_size=31)


def test_duplicate_text_groups() -> None:
    contracts = make_contracts(10, duplicates={1: 0, 5: 4, 6: 4})
    assert duplicate_text_groups(contracts) == [["c000", "c001"], ["c004", "c005", "c006"]]
