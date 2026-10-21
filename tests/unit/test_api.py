"""API behavior that needs no database: validation, limits, filenames and error shapes.

The session dependency yields a bare ``object()``, so any request that reached the database
would fail with a 500 instead of the expected 4xx: validation must happen first.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from retrieval_labs.api.app import create_app, get_session
from retrieval_labs.api.errors import ApiError
from retrieval_labs.api.uploads import clean_title, safe_filename
from retrieval_labs.config import Settings
from retrieval_labs.evaluation.report_tables import TEST_RUN, load_rows

ROOT = Path(__file__).resolve().parents[2]
SECRET = "secret-token-123"


class BrokenSessions:
    def __call__(self):
        raise RuntimeError(f"connection refused for password {SECRET}")


class BrokenQueue:
    class connection:  # noqa: N801
        @staticmethod
        def ping() -> None:
            raise ConnectionError(SECRET)


@asynccontextmanager
async def offline(app: FastAPI) -> AsyncIterator[None]:
    app.state.sessions = BrokenSessions()
    app.state.queue = BrokenQueue()
    yield


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(database_url="unused", redis_url="unused", max_upload_bytes=1000)
    app = create_app(settings, lifespan=offline, reports_dir=ROOT / "reports" / "benchmark")
    app.dependency_overrides[get_session] = lambda: iter([object()])
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def upload(client: TestClient, name: str, data: bytes, content_type: str = "text/plain"):
    return client.post("/contracts", files={"file": (name, data, content_type)})


def assert_error(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


@pytest.mark.parametrize("name", ["contract.pdf", "contract.docx", "contract", "txt"])
def test_only_txt_files_are_accepted(client: TestClient, name: str) -> None:
    assert_error(upload(client, name, b"text"), 415, "unsupported_file_type")


def test_a_txt_name_with_another_media_type_is_rejected(client: TestClient) -> None:
    assert_error(upload(client, "a.txt", b"text", "application/pdf"), 415, "unsupported_file_type")


def test_files_over_the_limit_are_rejected(client: TestClient) -> None:
    assert_error(upload(client, "a.txt", b"x" * 1001), 413, "file_too_large")


def test_huge_bodies_are_rejected_before_parsing(client: TestClient) -> None:
    assert_error(upload(client, "a.txt", b"x" * 200_000), 413, "file_too_large")


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"\xff\xfe not utf-8", "invalid_encoding"),
        (b"Clause one.\x00Clause two.", "invalid_text"),
        (b"", "empty_file"),
        (b" \n\t\r\n", "empty_file"),
    ],
)
def test_invalid_text_is_rejected(client: TestClient, data: bytes, code: str) -> None:
    assert_error(upload(client, "a.txt", data), 422, code)


def test_missing_file_field_is_a_validation_error(client: TestClient) -> None:
    response = client.post("/contracts", data={"title": "no file"})
    assert_error(response, 422, "invalid_request")
    assert "no file" not in response.text  # the input is not echoed back


def test_invalid_document_id_is_a_validation_error(client: TestClient) -> None:
    assert_error(client.get("/contracts/not-a-uuid/status"), 422, "invalid_request")


def test_unknown_routes_use_the_error_shape(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.put("/benchmarks/latest"), 405, "method_not_allowed")


def test_internal_errors_expose_no_details(client: TestClient) -> None:
    client.app.dependency_overrides.pop(get_session)
    response = client.get("/contracts/00000000-0000-0000-0000-000000000000/status")
    assert_error(response, 500, "internal_error")
    assert SECRET not in response.text
    assert "Traceback" not in response.text


def test_health_reports_unavailable_dependencies(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "unavailable",
        "queue": "unavailable",
    }
    assert SECRET not in response.text


def test_latest_benchmark_is_the_stored_test_run(client: TestClient) -> None:
    response = client.get("/benchmarks/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["run"] == TEST_RUN
    expected = load_rows(TEST_RUN, ROOT / "reports" / "benchmark")
    assert body["rows"] == [row.model_dump(mode="json") for row in expected]


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("contract.txt", "contract.txt"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\x\\..\\evil.txt", "evil.txt"),
        ("..\\..\\.hidden.txt", "hidden.txt"),
        ("a\x00b\nc\r.txt", "a_b_c_.txt"),
        ("<script>alert(1)</script>.txt", "script_.txt"),  # "/" splits the path
        ("Ｃｏｎｔｒａｃｔ.txt", "Contract.txt"),
        ("...", "upload.txt"),
        ("", "upload.txt"),
        (None, "upload.txt"),
    ],
)
def test_filenames_become_safe_labels(raw: str | None, safe: str) -> None:
    assert safe_filename(raw) == safe


def test_long_filenames_are_truncated() -> None:
    assert len(safe_filename("a" * 1000 + ".txt")) == 255


def test_title_defaults_to_the_filename_stem_and_rejects_control_characters() -> None:
    assert clean_title(None, "supply.txt") == "supply"
    assert clean_title("  Supply Agreement ", "x.txt") == "Supply Agreement"
    with pytest.raises(ApiError):
        clean_title("bad\x07title", "x.txt")
    with pytest.raises(ApiError):
        clean_title("t" * 256, "x.txt")


def test_importing_the_api_does_not_load_gold_annotations_or_torch() -> None:
    code = (
        "import sys, retrieval_labs.api.app; "
        "print('retrieval_labs.cuad.gold' in sys.modules, 'torch' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.split() == ["False", "False"]
