"""Download the two pinned models into the Hugging Face cache (``HF_HOME``), checking first.

    python scripts/download_models.py --check   # metadata only: revisions and sizes
    python scripts/download_models.py           # check, then download exactly these files

Before downloading, the Hugging Face API must confirm each pinned commit and report exactly
the expected size for every file; otherwise nothing is downloaded. The file lists are the
ones sentence-transformers used locally (Phase 4), so optional extras (ONNX, OpenVINO, other
framework weights) are never fetched. Afterwards each file's size is checked on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, constants, snapshot_download

from retrieval_labs.retrieval.embeddings import BGE_SMALL, ModelSpec
from retrieval_labs.retrieval.reranker import MINILM_RERANKER

# Byte sizes of the files loaded during Phase 4 (local cache, 2026-10-21).
EXPECTED: dict[ModelSpec, dict[str, int]] = {
    BGE_SMALL: {
        "1_Pooling/config.json": 190,
        "README.md": 94_783,
        "config.json": 743,
        "config_sentence_transformers.json": 124,
        "model.safetensors": 133_466_304,
        "modules.json": 349,
        "sentence_bert_config.json": 52,
        "special_tokens_map.json": 125,
        "tokenizer.json": 711_396,
        "tokenizer_config.json": 366,
        "vocab.txt": 231_508,
    },
    MINILM_RERANKER: {
        "config.json": 794,
        "model.safetensors": 90_870_598,
        "special_tokens_map.json": 132,
        "tokenizer.json": 711_396,
        "tokenizer_config.json": 1_330,
        "vocab.txt": 231_508,
    },
}


def check(api: HfApi) -> list[str]:
    problems: list[str] = []
    for spec, files in EXPECTED.items():
        info = api.model_info(spec.repo, revision=spec.revision, files_metadata=True)
        if info.sha != spec.revision:
            problems.append(f"{spec.repo}: API resolved {info.sha}, pinned {spec.revision}")
        remote = {s.rfilename: s.size for s in info.siblings or []}
        for name, size in files.items():
            if remote.get(name) != size:
                problems.append(f"{spec.repo}/{name}: remote {remote.get(name)} B, expected {size}")
        total = sum(files.values())
        print(f"{spec.repo}@{spec.revision[:8]}: {len(files)} files, {total / 2**20:.1f} MiB")
    grand = sum(sum(files.values()) for files in EXPECTED.values())
    print(f"total: {grand:,} bytes ({grand / 2**20:.1f} MiB)")
    return problems


def download() -> None:
    for spec, files in EXPECTED.items():
        path = Path(
            snapshot_download(spec.repo, revision=spec.revision, allow_patterns=list(files))
        )
        for name, size in files.items():
            actual = (path / name).stat().st_size
            if actual != size:
                sys.exit(f"{spec.repo}/{name}: {actual} B on disk, expected {size}")
        print(f"downloaded and verified {spec.repo}@{spec.revision[:8]} -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify metadata, download nothing")
    args = parser.parse_args()
    if constants.HF_HUB_OFFLINE:
        # The image is offline for the API and worker; Compose's models step sets it to 0.
        print(
            "HF_HUB_OFFLINE is set: run with HF_HUB_OFFLINE=0 to check or download", file=sys.stderr
        )
        return 2
    problems = check(HfApi())
    if problems:
        print("NOT downloading; metadata differs from the pinned expectation:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if args.check:
        print("check passed; nothing downloaded (--check)")
        return 0
    download()
    return 0


if __name__ == "__main__":
    sys.exit(main())
