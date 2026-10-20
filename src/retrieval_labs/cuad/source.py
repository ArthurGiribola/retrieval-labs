"""Download the official CUAD v1 archive and verify it against a pinned SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from retrieval_labs.cuad import CuadDataError

CUAD_URL = "https://github.com/The-Atticus-Project/cuad/raw/main/data.zip"
CUAD_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
DEFAULT_ARCHIVE_PATH = Path("data/raw/cuad/data.zip")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, expected_sha256: str = CUAD_SHA256) -> None:
    """Raise ``CuadDataError`` if the file is missing or its SHA-256 differs."""
    if not path.is_file():
        raise CuadDataError(f"archive not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise CuadDataError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def ensure_archive(
    path: Path = DEFAULT_ARCHIVE_PATH,
    url: str = CUAD_URL,
    expected_sha256: str = CUAD_SHA256,
) -> Path:
    """Return a verified archive at ``path``, downloading it only if absent.

    An existing file with the wrong checksum is never overwritten: the caller must
    inspect and remove it. Downloads go to a ``.part`` file that is renamed only
    after verification, so an interrupted run never leaves a "valid-looking" archive.
    """
    if path.exists():
        verify_archive(path, expected_sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        verify_archive(partial, expected_sha256)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path
