"""JSON log lines: one object per line with the call site's extra fields."""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from retrieval_labs.logging_setup import configure_logging


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    buffer = io.StringIO()
    configure_logging("INFO", buffer)
    yield buffer
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])


def lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().split("\n") if line]


def test_records_are_json_with_extra_fields(stream: io.StringIO) -> None:
    logging.getLogger("retrieval_labs.test").info(
        "job completed", extra={"job_id": "j1", "timings_s": {"total": 0.5}}
    )
    [entry] = lines(stream)
    assert entry["event"] == "job completed"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "retrieval_labs.test"
    assert entry["job_id"] == "j1"
    assert entry["timings_s"] == {"total": 0.5}
    assert entry["time"].endswith("+00:00")


def test_exceptions_are_kept_in_one_line(stream: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("retrieval_labs.test").exception("job failed")
    [entry] = lines(stream)
    assert "ValueError: boom" in entry["exception"]
    assert stream.getvalue().count("\n") == 1


def test_library_loggers_go_through_the_same_handler(stream: io.StringIO) -> None:
    logging.getLogger("uvicorn.access").warning("GET /health 200")
    logging.getLogger("rq.worker").warning("Job OK")
    assert [e["logger"] for e in lines(stream)] == ["uvicorn.access", "rq.worker"]
