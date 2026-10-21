"""JSON log lines for the API and the worker: one object per line, on stdout.

Fields: ``time``, ``level``, ``logger``, ``event`` (the message) and any ``extra`` given at
the call site (ids, hashes, sizes, durations). Contract text is never passed to a logger;
tests run whole jobs and check that no log line contains it.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO, Any

# Attributes every LogRecord has; anything else on a record came from ``extra``.
_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=True)


def configure_logging(level: str = "INFO", stream: IO[str] | None = None) -> None:
    """Send every logger (ours, uvicorn's, RQ's, SQLAlchemy's) through one JSON handler."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "rq.worker", "rq.job"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
