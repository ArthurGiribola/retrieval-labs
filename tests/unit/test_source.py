"""Download/verification tests. They use local ``file://`` URLs, never the network."""

import hashlib
from pathlib import Path

import pytest

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.source import ensure_archive, sha256_file, verify_archive

PAYLOAD = b"fake archive bytes"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def remote(tmp_path: Path) -> str:
    source = tmp_path / "remote" / "data.zip"
    source.parent.mkdir()
    source.write_bytes(PAYLOAD)
    return source.as_uri()


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(PAYLOAD)
    assert sha256_file(path) == PAYLOAD_SHA256


def test_verify_archive_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CuadDataError, match="not found"):
        verify_archive(tmp_path / "absent.zip", PAYLOAD_SHA256)


def test_ensure_archive_downloads_and_verifies(tmp_path: Path, remote: str) -> None:
    target = tmp_path / "raw" / "data.zip"
    assert ensure_archive(target, remote, PAYLOAD_SHA256) == target
    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path / "raw" / "data.zip.part").exists()


def test_ensure_archive_rejects_bad_download_and_leaves_nothing(
    tmp_path: Path, remote: str
) -> None:
    target = tmp_path / "raw" / "data.zip"
    with pytest.raises(CuadDataError, match="mismatch"):
        ensure_archive(target, remote, "0" * 64)
    assert not target.exists()
    assert not (tmp_path / "raw" / "data.zip.part").exists()


def test_ensure_archive_reuses_valid_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "data.zip"
    target.write_bytes(PAYLOAD)
    # An unreachable URL proves no download is attempted.
    assert ensure_archive(target, "file:///nonexistent/data.zip", PAYLOAD_SHA256) == target


def test_ensure_archive_never_overwrites_corrupt_existing_file(tmp_path: Path, remote: str) -> None:
    target = tmp_path / "data.zip"
    target.write_bytes(b"corrupted")
    with pytest.raises(CuadDataError, match="mismatch"):
        ensure_archive(target, remote, PAYLOAD_SHA256)
    assert target.read_bytes() == b"corrupted"
