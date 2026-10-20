"""Read and write the prepared CUAD dataset (JSONL files plus a manifest).

Contracts and gold annotations are stored in separate files. Loaders verify each file
against the SHA-256 recorded in the manifest, so stale or hand-edited data is rejected.
Gold annotations for the test split require an explicit ``final_evaluation=True``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.gold import GoldAnnotation
from retrieval_labs.cuad.source import sha256_file
from retrieval_labs.cuad.splits import Split
from retrieval_labs.documents import Contract

DEFAULT_PROCESSED_DIR = Path("data/processed/cuad")
CONTRACTS_FILE = "contracts.jsonl"
GOLD_FILE = "gold_annotations.jsonl"
STATS_FILE = "stats.json"
MANIFEST_FILE = "manifest.json"


class OutputFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    records: int


class Manifest(BaseModel):
    """Provenance and split membership of a prepared dataset. Contains no timestamps,
    so preparing the same archive twice yields byte-identical files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    json_version: str
    source_url: str
    archive_sha256: str
    generator_version: str
    labels: list[str]
    dev_size: int
    dev_seed: int
    split_counts: dict[str, int]
    duplicate_text_groups: list[list[str]]
    splits: dict[str, Split]
    files: dict[str, OutputFile]


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")
            count += 1
    return count


def read_manifest(directory: Path = DEFAULT_PROCESSED_DIR) -> Manifest:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise CuadDataError(f"no manifest at {path}; run scripts/prepare_cuad.py first")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _verified_lines(directory: Path, name: str, manifest: Manifest) -> list[str]:
    path = directory / name
    if sha256_file(path) != manifest.files[name].sha256:
        raise CuadDataError(f"{path} does not match the manifest; re-run preparation")
    return path.read_text(encoding="utf-8").splitlines()


def load_contracts(split: Split, directory: Path = DEFAULT_PROCESSED_DIR) -> list[Contract]:
    """Source contracts of one split. Safe to pass to inference code."""
    manifest = read_manifest(directory)
    contracts = [
        Contract.model_validate_json(line)
        for line in _verified_lines(directory, CONTRACTS_FILE, manifest)
    ]
    return [c for c in contracts if manifest.splits[c.contract_id] == split]


def load_gold(
    split: Split,
    directory: Path = DEFAULT_PROCESSED_DIR,
    *,
    final_evaluation: bool = False,
) -> list[GoldAnnotation]:
    """Gold annotations of one split. EVALUATION ONLY.

    The test split is locked unless ``final_evaluation=True``, so that tuning code
    cannot read test labels by accident.
    """
    if split == Split.TEST and not final_evaluation:
        raise CuadDataError(
            "test gold annotations are reserved for final evaluation; "
            "tune on the dev split or pass final_evaluation=True deliberately"
        )
    manifest = read_manifest(directory)
    gold = [
        GoldAnnotation.model_validate_json(line)
        for line in _verified_lines(directory, GOLD_FILE, manifest)
    ]
    return [g for g in gold if manifest.splits[g.contract_id] == split]
