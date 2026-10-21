"""Worker configuration: the extractor follows the settings; the paid one is capped."""

import pytest

from retrieval_labs.config import ConfigError, Settings, load_settings
from retrieval_labs.extraction.llm import FakeExtractor
from retrieval_labs.worker.main import build_extractor, main

URLS = {"DATABASE_URL": "postgresql+psycopg://u:p@h/db", "REDIS_URL": "redis://h/0"}


def test_fake_is_free_and_has_no_ledger() -> None:
    extractor, ledger = build_extractor(Settings(database_url="x", redis_url="y"))
    assert isinstance(extractor, FakeExtractor)
    assert ledger is None


def test_anthropic_cannot_be_configured_without_a_spending_cap() -> None:
    with pytest.raises(ConfigError, match="LLM_MAX_USD"):
        load_settings({**URLS, "LLM_PROVIDER": "anthropic"})


def test_the_worker_exits_on_invalid_settings_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert main([]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err
