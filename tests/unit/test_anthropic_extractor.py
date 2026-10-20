"""AnthropicExtractor tests with a stand-in client: no network, no API key, no cost."""

from types import SimpleNamespace
from typing import Any

import pytest

from retrieval_labs.clauses import ClauseType
from retrieval_labs.extraction.anthropic_extractor import AnthropicExtractor
from retrieval_labs.extraction.evidence import Passage
from retrieval_labs.extraction.llm import build_request
from retrieval_labs.extraction.prompt import SYSTEM_PROMPT
from retrieval_labs.extraction.schema import GoverningLawAnswer, Status

VALID = (
    '{"status": "FOUND", "quotes": [{"passage_id": "P1", "text": "laws of New York"}], '
    '"notes": "", "value": {"jurisdiction": "New York"}}'
)


class StubMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def stub_client(text: str | None, stop_reason: str = "end_turn", **extra: Any) -> Any:
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    usage = SimpleNamespace(
        input_tokens=2_000,
        output_tokens=1_000,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    response = SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, **extra)
    return SimpleNamespace(messages=StubMessages(response))


def request():
    passage = Passage("P1", 0, 30, "Governed by laws of New York.", (("c#0000", 0, 30),))
    return build_request(ClauseType.GOVERNING_LAW, [passage])


EXPECTED_COST = (2_000 * 0.10 + 1_000 * 0.50) / 1_000_000


def test_request_shape() -> None:
    client = stub_client(VALID)
    AnthropicExtractor(client=client).extract(request())
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] == 4000
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"][0]["role"] == "user"
    assert call["output_config"]["effort"] == "medium"
    schema = call["output_config"]["format"]["schema"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert schema["additionalProperties"] is False
    # Haiku 5.5 rejects non-default sampling parameters; none may be sent.
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(call)


def test_valid_answer_is_parsed_with_usage_and_cost() -> None:
    response = AnthropicExtractor(client=stub_client(VALID)).extract(request())
    assert isinstance(response.answer, GoverningLawAnswer)
    assert response.answer.status == Status.FOUND
    assert response.usage.input_tokens == 2_000
    assert response.usage.cost_usd == pytest.approx(EXPECTED_COST)
    assert response.failure is None


@pytest.mark.parametrize(
    ("text", "stop_reason", "extra", "reason"),
    [
        ('{"status": "FOU', "max_tokens", {}, "incomplete response"),
        (None, "refusal", {"stop_details": SimpleNamespace(category="cyber")}, "category=cyber"),
        ('{"status": "FOUND"}', "end_turn", {}, "did not match the schema"),
        (None, "end_turn", {}, "no text block"),
    ],
)
def test_failures_keep_usage_so_cost_is_never_lost(text, stop_reason, extra, reason) -> None:
    client = stub_client(text, stop_reason, **extra)
    response = AnthropicExtractor(client=client).extract(request())
    assert response.answer is None
    assert reason in response.failure
    assert response.usage.cost_usd == pytest.approx(EXPECTED_COST)


def test_worst_case_cost_uses_max_tokens() -> None:
    extractor = AnthropicExtractor(client=stub_client(VALID), max_tokens=4000)
    assert extractor.worst_case_cost_usd(request()) >= 4000 * 0.50 / 1_000_000


def test_unknown_model_is_rejected_before_any_call() -> None:
    with pytest.raises(ValueError, match="no price card"):
        AnthropicExtractor(model="claude-unknown", client=stub_client(VALID))
