"""Settings: read once from the environment, validated, never echoing secret values."""

from pathlib import Path

import pytest

from retrieval_labs.config import ConfigError, Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]

URLS = {
    "DATABASE_URL": "postgresql+psycopg://user:s3cret-pw@db:5432/retrieval_labs",
    "REDIS_URL": "redis://redis:6379/0",
}


def test_defaults_are_the_safe_and_measured_ones() -> None:
    settings = load_settings(URLS)
    assert settings.retriever == "hybrid+rerank"  # D-001
    assert settings.llm_provider == "fake"
    assert settings.llm_max_usd is None
    assert settings.api_host == "127.0.0.1"
    assert settings.job_timeout_s == 900
    assert settings.lost_after.total_seconds() == 900 + 120


def test_values_are_parsed_and_blank_variables_fall_back_to_defaults() -> None:
    settings = load_settings(
        {**URLS, "RETRIEVER": "bm25", "JOB_TIMEOUT_S": "60", "LLM_MAX_USD": "", "LOG_LEVEL": " "}
    )
    assert (settings.retriever, settings.job_timeout_s) == ("bm25", 60)
    assert settings.llm_max_usd is None
    assert settings.log_level == "INFO"


def test_the_paid_provider_needs_a_positive_cap() -> None:
    assert load_settings({**URLS, "LLM_PROVIDER": "anthropic", "LLM_MAX_USD": "0.25"}).llm_max_usd
    with pytest.raises(ConfigError, match="LLM_MAX_USD"):
        load_settings({**URLS, "LLM_PROVIDER": "anthropic", "LLM_MAX_USD": "0"})


@pytest.mark.parametrize(
    ("overrides", "variable"),
    [
        ({"DATABASE_URL": ""}, "DATABASE_URL"),
        ({"RETRIEVER": "magic"}, "RETRIEVER"),
        ({"LLM_PROVIDER": "openai"}, "LLM_PROVIDER"),
        ({"JOB_TIMEOUT_S": "-5"}, "JOB_TIMEOUT_S"),
        ({"MAX_UPLOAD_BYTES": "lots"}, "MAX_UPLOAD_BYTES"),
        ({"API_PORT": "70000"}, "API_PORT"),
    ],
)
def test_invalid_settings_name_the_variable(overrides: dict, variable: str) -> None:
    with pytest.raises(ConfigError, match=variable):
        load_settings({**URLS, **overrides})


def test_errors_never_echo_values() -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({**URLS, "REDIS_URL": "", "JOB_TIMEOUT_S": "s3cret-pw"})
    assert "s3cret-pw" not in str(caught.value)


def test_env_example_documents_every_setting() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").split("\n")
    names = {line.split("=", 1)[0] for line in lines if "=" in line and not line.startswith("#")}
    assert {name.upper() for name in Settings.model_fields} <= names
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    load_settings(dict(pairs))  # the example itself is a valid configuration
