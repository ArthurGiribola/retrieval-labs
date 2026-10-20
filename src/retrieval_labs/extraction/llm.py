"""Provider-neutral extraction interface, pricing, and a deterministic fake provider.

Application code depends on ``Extractor`` only; no vendor SDK is imported here.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from retrieval_labs.clauses import ClauseType
from retrieval_labs.extraction.evidence import Passage
from retrieval_labs.extraction.prompt import SYSTEM_PROMPT, build_user_message
from retrieval_labs.extraction.schema import (
    ANSWER_MODELS,
    VALUE_MODELS,
    ClauseAnswer,
    LLMUsage,
    Status,
)


@dataclass(frozen=True)
class ExtractionRequest:
    clause_type: ClauseType
    system: str
    user: str
    answer_model: type[ClauseAnswer]


@dataclass(frozen=True)
class ExtractionResponse:
    answer: ClauseAnswer | None
    stop_reason: str
    usage: LLMUsage
    raw_text: str | None = None
    failure: str | None = None  # why there is no answer, when answer is None


class Extractor(Protocol):
    provider: str
    model: str
    settings: dict[str, Any]

    def worst_case_cost_usd(self, request: ExtractionRequest) -> float: ...

    def extract(self, request: ExtractionRequest) -> ExtractionResponse: ...


def build_request(clause_type: ClauseType, passages: Sequence[Passage]) -> ExtractionRequest:
    return ExtractionRequest(
        clause_type=clause_type,
        system=SYSTEM_PROMPT,
        user=build_user_message(clause_type, passages),
        answer_model=ANSWER_MODELS[clause_type],
    )


# --- Pricing ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    max_prompt_tokens: int  # this rate card applies only up to this prompt size


# Checked 2026-10-20 at https://platform.claude.com/docs/en/about-claude/pricing
PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(
        input=0.10, output=0.50, cache_read=0.01, cache_write_5m=0.125, max_prompt_tokens=100_000
    ),
}

# Fixed allowance for request framing the character count cannot see.
_INPUT_OVERHEAD_TOKENS = 1_000


def cost_usd(price: Price, usage: LLMUsage) -> float:
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write_5m
    ) / 1_000_000


def input_token_bound(request: ExtractionRequest) -> int:
    """Upper bound on input tokens, assuming at most one token per character (checked
    against measured usage in the pilot; see ENGINEERING_LOG) plus a fixed overhead."""
    schema = json.dumps(request.answer_model.model_json_schema())
    return len(request.system) + len(request.user) + len(schema) + _INPUT_OVERHEAD_TOKENS


def worst_case_cost_usd(price: Price, request: ExtractionRequest, max_tokens: int) -> float:
    bound = input_token_bound(request)
    if bound > price.max_prompt_tokens:
        raise ValueError(f"prompt bound {bound} exceeds the {price.max_prompt_tokens}-token card")
    return (bound * price.input + max_tokens * price.output) / 1_000_000


# --- Fake provider ------------------------------------------------------------------------


def empty_answer(clause_type: ClauseType, status: Status = Status.NOT_FOUND) -> ClauseAnswer:
    value_model = VALUE_MODELS[clause_type]
    value = {
        name: [] if typing.get_origin(info.annotation) is list else None
        for name, info in value_model.model_fields.items()
    }
    return ANSWER_MODELS[clause_type].model_validate(
        {"status": status, "quotes": [], "notes": "", "value": value}
    )


Responder = Callable[[ExtractionRequest], ClauseAnswer | None]


@dataclass
class FakeExtractor:
    """Deterministic, free extractor for tests and dry runs. By default answers NOT_FOUND."""

    responder: Responder | None = None
    provider: str = "fake"
    model: str = "fake"
    settings: dict[str, Any] = field(default_factory=dict)
    requests: list[ExtractionRequest] = field(default_factory=list)

    def worst_case_cost_usd(self, request: ExtractionRequest) -> float:
        return 0.0

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        self.requests.append(request)
        if self.responder is None:
            answer: ClauseAnswer | None = empty_answer(request.clause_type)
        else:
            answer = self.responder(request)
        return ExtractionResponse(
            answer=answer,
            stop_reason="end_turn",
            usage=LLMUsage(),
            failure=None if answer else "fake responder returned no answer",
        )
