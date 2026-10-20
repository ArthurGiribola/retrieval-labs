import json
from pathlib import Path

import pytest

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.parse import make_contract_id
from retrieval_labs.cuad.prepare import prepare_dataset
from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import (
    CONTRACTS_FILE,
    GOLD_FILE,
    MANIFEST_FILE,
    STATS_FILE,
    load_contracts,
    load_gold,
    read_manifest,
)
from synthetic_cuad import ALPHA, DELTA, SYNTHETIC

OUTPUTS = [CONTRACTS_FILE, GOLD_FILE, STATS_FILE, MANIFEST_FILE]


@pytest.fixture
def prepared(cuad_archive: Path, tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed"
    prepare_dataset(cuad_archive, out_dir, dev_size=1, seed=42)
    return out_dir


def test_writes_all_outputs_with_manifest(prepared: Path) -> None:
    manifest = read_manifest(prepared)
    assert all((prepared / name).is_file() for name in OUTPUTS)
    assert manifest.files[CONTRACTS_FILE].records == len(SYNTHETIC)
    assert manifest.files[GOLD_FILE].records == len(SYNTHETIC) * len(CuadLabel)
    assert manifest.split_counts == {"dev": 1, "test": 1, "train": 2}
    assert manifest.splits[make_contract_id(DELTA)] == Split.TEST


def test_preparation_is_byte_reproducible(cuad_archive: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    prepare_dataset(cuad_archive, first, dev_size=1, seed=42)
    prepare_dataset(cuad_archive, second, dev_size=1, seed=42)
    for name in OUTPUTS:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_files_use_lf_line_endings(prepared: Path) -> None:
    for name in OUTPUTS:
        assert b"\r\n" not in (prepared / name).read_bytes(), name


def test_files_are_pure_ascii(prepared: Path) -> None:
    """Non-ASCII text (including U+2028) must be escaped so no reader can split a record."""
    for name in OUTPUTS:
        assert (prepared / name).read_bytes().isascii(), name


def test_jsonl_has_one_line_per_record(prepared: Path) -> None:
    manifest = read_manifest(prepared)
    for name in [CONTRACTS_FILE, GOLD_FILE]:
        text = (prepared / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) == manifest.files[name].records, name


def test_load_contracts_by_split(prepared: Path) -> None:
    manifest = read_manifest(prepared)
    for split in Split:
        loaded = load_contracts(split, prepared)
        expected = {cid for cid, s in manifest.splits.items() if s == split}
        assert {c.contract_id for c in loaded} == expected


def test_round_trip_preserves_text_exactly(prepared: Path) -> None:
    contracts = [c for split in Split for c in load_contracts(split, prepared)]
    by_title = {c.title: c.text for c in contracts}
    assert by_title == {title: text for title, (text, _) in SYNTHETIC.items()}


def test_test_gold_is_locked_without_final_evaluation(prepared: Path) -> None:
    with pytest.raises(CuadDataError, match="final evaluation"):
        load_gold(Split.TEST, prepared)
    unlocked = load_gold(Split.TEST, prepared, final_evaluation=True)
    assert {g.contract_id for g in unlocked} == {make_contract_id(DELTA)}


def test_train_and_dev_gold_need_no_unlock(prepared: Path) -> None:
    train_and_dev = load_gold(Split.TRAIN, prepared) + load_gold(Split.DEV, prepared)
    assert len(train_and_dev) == 3 * len(CuadLabel)


def test_loaders_reject_modified_files(prepared: Path) -> None:
    path = prepared / CONTRACTS_FILE
    path.write_bytes(path.read_bytes().replace(b"New York", b"New Jersey"))
    with pytest.raises(CuadDataError, match="does not match the manifest"):
        load_contracts(Split.TRAIN, prepared)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CuadDataError, match="no manifest"):
        load_contracts(Split.DEV, tmp_path)


def test_stats_count_multi_span_label(prepared: Path) -> None:
    stats = json.loads((prepared / STATS_FILE).read_text(encoding="utf-8"))
    alpha_split = read_manifest(prepared).splits[make_contract_id(ALPHA)]
    cap = stats["labels"]["Cap On Liability"][alpha_split]
    assert cap["contracts_with_answer"] == 1
    assert cap["contracts_with_multiple_spans"] == 1
    assert cap["total_spans"] == 2
    assert cap["overlapping_span_pairs"] == 0
