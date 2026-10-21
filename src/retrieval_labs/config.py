"""Application settings for the API and the worker, read once from the environment.

Every variable is listed in ``.env.example``. Invalid settings stop the process at startup
with the names of the offending variables; values are never echoed, since some (the
database URL) contain credentials.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import timedelta
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from retrieval_labs.retrieval.factory import APPLICATION_METHOD

RetrieverName = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]
# A job still RUNNING this long after its time limit lost its worker (crash, kill, restart).
LOST_JOB_GRACE = timedelta(minutes=2)
DEFAULT_JOB_TIMEOUT_S = 900


class ConfigError(RuntimeError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    database_url: str = Field(min_length=1)
    redis_url: str = Field(min_length=1)
    retriever: RetrieverName = APPLICATION_METHOD
    llm_provider: Literal["fake", "anthropic"] = "fake"
    # Spending cap for one worker process's lifetime; required for the paid provider.
    llm_max_usd: float | None = Field(default=None, gt=0)
    job_timeout_s: int = Field(default=DEFAULT_JOB_TIMEOUT_S, gt=0)
    max_upload_bytes: int = Field(default=2_000_000, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0, lt=65536)

    @model_validator(mode="after")
    def _paid_provider_needs_a_cap(self) -> Self:
        if self.llm_provider == "anthropic" and self.llm_max_usd is None:
            raise ValueError("LLM_PROVIDER=anthropic requires LLM_MAX_USD (spending cap)")
        return self

    @property
    def lost_after(self) -> timedelta:
        return timedelta(seconds=self.job_timeout_s) + LOST_JOB_GRACE


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Settings from environment variables named like the fields, upper-cased. Empty
    variables count as unset, so a blank line in ``.env`` falls back to the default."""
    environ = os.environ if environ is None else environ
    values = {
        name: environ[name.upper()]
        for name in Settings.model_fields
        if environ.get(name.upper(), "").strip()
    }
    try:
        return Settings.model_validate(values)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']).upper() or 'settings'}: {e['msg']}"
            for e in error.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid configuration: {problems}") from None
