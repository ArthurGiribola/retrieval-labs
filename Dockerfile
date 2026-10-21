# One image for the API, the worker, migrations and the model download; Compose picks the
# command. Linux resolves torch from the PyTorch CPU index (LOG-016); the build fails if any
# CUDA package is installed. Model weights are not in the image: they live in a volume.

FROM ghcr.io/astral-sh/uv:0.13.2 AS uv

FROM python:3.12.14-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group demo --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group demo --no-editable
COPY docker/check_cpu_only.py /tmp/check_cpu_only.py
RUN /app/.venv/bin/python /tmp/check_cpu_only.py

FROM python:3.12.14-slim-trixie
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models /app/data/cache \
    && chown app:app /models /app/data /app/data/cache
COPY --from=build --chown=app:app /app/.venv /app/.venv
WORKDIR /app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app reports/benchmark ./reports/benchmark
COPY --chown=app:app scripts/download_models.py ./scripts/
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1
USER app
CMD ["python", "-m", "retrieval_labs.api"]
