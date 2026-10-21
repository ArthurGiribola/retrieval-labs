"""Run the API: ``uv run python -m retrieval_labs.api``.

Binds ``API_HOST`` (default 127.0.0.1). Only Docker sets 0.0.0.0, inside the container,
with the port published on the host's loopback interface alone.
"""

from __future__ import annotations

import sys

import uvicorn

from retrieval_labs.api.app import create_app
from retrieval_labs.config import ConfigError, load_settings
from retrieval_labs.logging_setup import configure_logging


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    # log_config=None keeps uvicorn from installing its own (non-JSON) handlers.
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
