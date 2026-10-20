"""Integrity checks on the real prepared CUAD dataset.

Skipped when the dataset has not been prepared locally
(``uv run python scripts/prepare_cuad.py --download``). Expected counts come from the
independent raw-file inspection in ``scripts/inspect_cuad.py`` (see docs/DATASET.md),
so these tests cross-check two separate implementations.
"""

import json
from pathlib import Path

import pytest

from retrieval_labs.cuad.gold import GoldAnnotation, check_offsets
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.parse import parse_archive
from retrieval_labs.cuad.source import CUAD_SHA256
from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import CONTRACTS_FILE, load_contracts, load_gold, read_manifest
from retrieval_labs.documents import Contract

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "cuad"
ARCHIVE = ROOT / "data" / "raw" / "cuad" / "data.zip"


@pytest.fixture(scope="module")
def contracts() -> dict[str, Contract]:
    if not (PROCESSED / "manifest.json").is_file():
        pytest.skip("prepared CUAD dataset not found")
    return {c.contract_id: c for split in Split for c in load_contracts(split, PROCESSED)}


@pytest.fixture(scope="module")
def gold(contracts: dict[str, Contract]) -> list[GoldAnnotation]:
    # Integrity checks are not tuning, so reading test gold here is legitimate.
    return [
        g
        for split in Split
        for g in load_gold(split, PROCESSED, final_evaluation=split == Split.TEST)
    ]


def test_manifest_provenance_and_split_sizes(contracts: dict[str, Contract]) -> None:
    manifest = read_manifest(PROCESSED)
    assert manifest.archive_sha256 == CUAD_SHA256
    assert (manifest.dev_size, manifest.dev_seed) == (102, 42)
    assert manifest.split_counts == {"train": 306, "dev": 102, "test": 102}
    assert len(contracts) == 510


def test_every_gold_span_matches_its_contract_text(
    contracts: dict[str, Contract], gold: list[GoldAnnotation]
) -> None:
    for annotation in gold:
        check_offsets(annotation, contracts[annotation.contract_id])


def test_gold_counts_match_independent_inspection(gold: list[GoldAnnotation]) -> None:
    assert len(gold) == 510 * len(CuadLabel)
    assert sum(g.is_answerable for g in gold) == 1182
    assert sum(len(g.spans) for g in gold) == 1714


def test_splits_follow_official_split_and_dev_comes_from_train(
    contracts: dict[str, Contract],
) -> None:
    if not ARCHIVE.is_file():
        pytest.skip("raw CUAD archive not found")
    parsed = parse_archive(ARCHIVE)
    splits = read_manifest(PROCESSED).splits
    test_ids = {cid for cid, s in splits.items() if s == Split.TEST}
    train_dev_ids = {cid for cid, s in splits.items() if s != Split.TEST}
    assert test_ids == parsed.official_test_ids
    assert train_dev_ids == parsed.official_train_ids


def test_identical_texts_share_a_split(contracts: dict[str, Contract]) -> None:
    manifest = read_manifest(PROCESSED)
    assert manifest.duplicate_text_groups, "expected the known duplicate pair"
    for group in manifest.duplicate_text_groups:
        assert len({manifest.splits[cid] for cid in group}) == 1


def test_contracts_file_has_no_annotation_fields(contracts: dict[str, Contract]) -> None:
    with (PROCESSED / CONTRACTS_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            assert set(json.loads(line)) == set(Contract.model_fields)
