"""Worker configuration: the paid provider cannot start without a spending cap."""

import pytest

from retrieval_labs.extraction.llm import FakeExtractor
from retrieval_labs.worker.main import build_extractor


def test_fake_is_free_and_has_no_ledger() -> None:
    extractor, ledger = build_extractor("fake")
    assert isinstance(extractor, FakeExtractor)
    assert ledger is None


def test_anthropic_requires_a_spending_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MAX_USD", raising=False)
    with pytest.raises(ValueError, match="LLM_MAX_USD"):
        build_extractor("anthropic")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_extractor("openai")
