"""Worker process: load the retriever and extractor once, then consume the analysis queue.

    uv run python -m retrieval_labs.worker.main [--burst]

Environment: ``DATABASE_URL``, ``REDIS_URL``, ``RETRIEVER`` (default ``hybrid+rerank``),
``LLM_PROVIDER`` (``fake``, the default, or ``anthropic``, which also needs ``LLM_MAX_USD``,
the spending cap for this worker's lifetime), ``JOB_TIMEOUT_S``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime

from redis import Redis
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty, UnixSignalDeathPenalty

from retrieval_labs.db.session import make_engine, make_session_factory
from retrieval_labs.extraction.budget import DEFAULT_LEDGER_PATH, SpendLedger
from retrieval_labs.extraction.llm import Extractor, FakeExtractor
from retrieval_labs.retrieval.factory import APPLICATION_METHOD, build_retrievers
from retrieval_labs.worker.queue import QUEUE_NAME
from retrieval_labs.worker.tasks import Runtime, configure, sweep_lost_jobs

logger = logging.getLogger(__name__)


class AnalysisWorker(SimpleWorker):
    """Runs jobs in the worker process itself (no fork per job), so models load once and
    torch never runs in a forked child. Windows has no SIGALRM, hence the timer there."""

    death_penalty_class = TimerDeathPenalty if sys.platform == "win32" else UnixSignalDeathPenalty


def build_extractor(provider: str) -> tuple[Extractor, SpendLedger | None]:
    if provider == "fake":
        return FakeExtractor(), None
    if provider != "anthropic":
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}; use 'fake' or 'anthropic'")
    max_usd = os.environ.get("LLM_MAX_USD")
    if not max_usd:
        raise ValueError("LLM_PROVIDER=anthropic requires LLM_MAX_USD (spending cap)")
    from retrieval_labs.extraction.anthropic_extractor import AnthropicExtractor

    run_id = datetime.now(UTC).strftime("worker-%Y%m%dT%H%M%SZ")
    return AnthropicExtractor(), SpendLedger(DEFAULT_LEDGER_PATH, run_id, float(max_usd))


def build_runtime() -> Runtime:
    provider = os.environ.get("LLM_PROVIDER", "fake")
    extractor, ledger = build_extractor(provider)
    method = os.environ.get("RETRIEVER", APPLICATION_METHOD)
    retrievers, info = build_retrievers([method])
    return Runtime(
        sessions=make_session_factory(make_engine()),
        retriever=retrievers[method],
        extractor=extractor,
        embedder=info.get("embedder"),
        ledger=ledger,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consume the analysis queue.")
    parser.add_argument("--burst", action="store_true", help="exit when the queue is empty")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    runtime = build_runtime()
    configure(runtime)
    sweep_lost_jobs(runtime)
    logger.info(
        "worker ready",
        extra={"retriever": runtime.retriever.name, "provider": runtime.extractor.provider},
    )
    redis = Redis.from_url(os.environ["REDIS_URL"])
    worker = AnalysisWorker([Queue(QUEUE_NAME, connection=redis)], connection=redis)
    worker.work(burst=args.burst, with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
