"""Claude extractor. The only module that imports the Anthropic SDK.

Uses ``messages.create`` with a JSON-schema ``output_config.format`` and validates the
answer itself, instead of ``messages.parse``: in anthropic 1.15, ``parse`` validates the
JSON inside the call, so a truncated (``max_tokens``) or refused response raises before
``usage`` can be read, and a billed call would escape the spend ledger (LOG-007).
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
from pydantic import ValidationError

from retrieval_labs.extraction.llm import (
    PRICES,
    ExtractionRequest,
    ExtractionResponse,
    cost_usd,
    worst_case_cost_usd,
)
from retrieval_labs.extraction.schema import LLMUsage

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicExtractor:
    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 4000,
        client: Any = None,
    ) -> None:
        if model not in PRICES:
            raise ValueError(f"no price card for {model!r}; add it to PRICES first")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.settings = {"effort": effort, "max_tokens": max_tokens}
        self._price = PRICES[model]
        # Credentials come from the environment (ANTHROPIC_API_KEY); never passed in code.
        self._client = client if client is not None else anthropic.Anthropic()

    def worst_case_cost_usd(self, request: ExtractionRequest) -> float:
        return worst_case_cost_usd(self._price, request, self.max_tokens)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        schema = anthropic.transform_schema(request.answer_model.model_json_schema())
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        latency = time.perf_counter() - started

        counts = LLMUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
        )
        usage = counts.model_copy(
            update={"cost_usd": cost_usd(self._price, counts), "latency_s": round(latency, 3)}
        )
        text = next((block.text for block in message.content if block.type == "text"), None)

        def failed(reason: str) -> ExtractionResponse:
            return ExtractionResponse(None, str(message.stop_reason), usage, text, reason)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            return failed(f"model declined (refusal, category={category})")
        if message.stop_reason != "end_turn":
            return failed(f"incomplete response (stop_reason={message.stop_reason})")
        if text is None:
            return failed("response contained no text block")
        try:
            answer = request.answer_model.model_validate_json(text)
        except ValidationError as exc:
            return failed(f"response did not match the schema: {exc.error_count()} error(s)")
        return ExtractionResponse(answer, str(message.stop_reason), usage, text)
