"""End-to-end dataset preparation: verified archive -> parsed data -> splits -> files."""

from __future__ import annotations

from pathlib import Path

import retrieval_labs
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.parse import parse_archive
from retrieval_labs.cuad.source import CUAD_URL, sha256_file
from retrieval_labs.cuad.splits import DEV_SEED, DEV_SIZE, assign_splits, duplicate_text_groups
from retrieval_labs.cuad.stats import compute_stats
from retrieval_labs.cuad.store import (
    CONTRACTS_FILE,
    GOLD_FILE,
    MANIFEST_FILE,
    STATS_FILE,
    Manifest,
    OutputFile,
    write_json,
    write_jsonl,
)


def prepare_dataset(
    archive: Path,
    out_dir: Path,
    dev_size: int = DEV_SIZE,
    seed: int = DEV_SEED,
    source_url: str = CUAD_URL,
) -> Manifest:
    """Prepare the dataset from an archive the caller has already verified.

    The manifest is written last: a run interrupted earlier leaves files whose
    hashes do not match any manifest, and the loaders refuse them.
    """
    parsed = parse_archive(archive)
    assignment = assign_splits(
        parsed.contracts, parsed.official_train_ids, parsed.official_test_ids, dev_size, seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_FILE).unlink(missing_ok=True)
    records = {
        CONTRACTS_FILE: write_jsonl(out_dir / CONTRACTS_FILE, parsed.contracts),
        GOLD_FILE: write_jsonl(out_dir / GOLD_FILE, parsed.gold),
    }
    write_json(out_dir / STATS_FILE, compute_stats(parsed.contracts, parsed.gold, assignment))
    records[STATS_FILE] = 1

    manifest = Manifest(
        dataset="CUAD v1",
        json_version=parsed.json_version,
        source_url=source_url,
        archive_sha256=sha256_file(archive),
        generator_version=retrieval_labs.__version__,
        labels=[label.value for label in CuadLabel],
        dev_size=dev_size,
        dev_seed=seed,
        split_counts={
            split.value: sum(1 for s in assignment.values() if s == split)
            for split in sorted(set(assignment.values()))
        },
        duplicate_text_groups=duplicate_text_groups(parsed.contracts),
        splits=assignment,
        files={
            name: OutputFile(sha256=sha256_file(out_dir / name), records=count)
            for name, count in records.items()
        },
    )
    write_json(out_dir / MANIFEST_FILE, manifest.model_dump(mode="json"))
    return manifest
