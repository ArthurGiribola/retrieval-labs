"""Worker process: load the retriever and extractor once, then consume the analysis queue.

    uv run python -m retrieval_labs.worker.main [--burst]

Settings come from the environment (``retrieval_labs.config``, listed in ``.env.example``).
``LLM_PROVIDER=anthropic`` is paid and refuses to start without ``LLM_MAX_USD``, the
spending cap for this worker's lifetime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from redis import Redis
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty, UnixSignalDeathPenalty

from retrieval_labs.config import ConfigError, Settings, load_settings
from retrieval_labs.db.session import make_engine, make_session_factory
from retrieval_labs.extraction.budget import DEFAULT_LEDGER_PATH, SpendLedger
from retrieval_labs.extraction.llm import Extractor, FakeExtractor
from retrieval_labs.logging_setup import configure_logging
from retrieval_labs.retrieval.factory import build_retrievers
from retrieval_labs.worker.queue import QUEUE_NAME
from retrieval_labs.worker.tasks import Runtime, configure, sweep_lost_jobs

logger = logging.getLogger("retrieval_labs.worker.main")  # not "__main__" under -m


class AnalysisWorker(SimpleWorker):
    """Runs jobs in the worker process itself (no fork per job), so models load once and
    torch never runs in a forked child. Windows has no SIGALRM, hence the timer there."""

    death_penalty_class = TimerDeathPenalty if sys.platform == "win32" else UnixSignalDeathPenalty


def build_extractor(settings: Settings) -> tuple[Extractor, SpendLedger | None]:
    if settings.llm_provider == "fake":
        return FakeExtractor(), None
    from retrieval_labs.extraction.anthropic_extractor import AnthropicExtractor

    # Settings guarantee the cap is set for the paid provider.
    run_id = datetime.now(UTC).strftime("worker-%Y%m%dT%H%M%SZ")
    ledger = SpendLedger(DEFAULT_LEDGER_PATH, run_id, settings.llm_max_usd)
    return AnthropicExtractor(), ledger


def build_runtime(settings: Settings) -> Runtime:
    extractor, ledger = build_extractor(settings)
    retrievers, info = build_retrievers([settings.retriever])
    return Runtime(
        sessions=make_session_factory(make_engine(settings.database_url)),
        retriever=retrievers[settings.retriever],
        extractor=extractor,
        embedder=info.get("embedder"),
        ledger=ledger,
        lost_after=settings.lost_after,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consume the analysis queue.")
    parser.add_argument("--burst", action="store_true", help="exit when the queue is empty")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    runtime = build_runtime(settings)
    configure(runtime)
    sweep_lost_jobs(runtime)
    logger.info(
        "worker ready",
        extra={
            "retriever": runtime.retriever.name,
            "provider": runtime.extractor.provider,
            "job_timeout_s": settings.job_timeout_s,
        },
    )
    redis = Redis.from_url(settings.redis_url)
    worker = AnalysisWorker([Queue(QUEUE_NAME, connection=redis)], connection=redis)
    worker.work(burst=args.burst, with_scheduler=False, logging_level=settings.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
