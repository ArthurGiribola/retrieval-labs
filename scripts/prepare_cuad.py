"""Prepare the CUAD v1 evaluation dataset.

Verifies (and, with --download, fetches) the official archive, then writes
contracts, gold annotations, statistics and a manifest to the output directory.

Usage:
    uv run python scripts/prepare_cuad.py [--download] [--archive PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.prepare import prepare_dataset
from retrieval_labs.cuad.source import CUAD_URL, DEFAULT_ARCHIVE_PATH, ensure_archive
from retrieval_labs.cuad.store import DEFAULT_PROCESSED_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--download", action="store_true", help=f"download the archive if absent ({CUAD_URL})"
    )
    args = parser.parse_args()

    try:
        if not args.archive.exists() and not args.download:
            raise CuadDataError(f"{args.archive} not found; re-run with --download")
        ensure_archive(args.archive)
        manifest = prepare_dataset(args.archive, args.out_dir)
    except CuadDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"prepared {manifest.dataset} into {args.out_dir}")
    print(f"  splits: {manifest.split_counts}")
    print(f"  duplicate-text groups: {manifest.duplicate_text_groups}")
    for name, info in manifest.files.items():
        print(f"  {name}: {info.records} records, sha256 {info.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
