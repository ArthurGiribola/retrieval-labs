"""Schemas for model answers (what the LLM returns) and verified results (what we report).

Answer models are sent to the LLM as structured-output schemas. Results are built by
our code from an answer plus verification against the contract text; the model never
writes offsets or the final ``extracted_text``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval_labs.clauses import SECTION_OF, ClauseType, ReportSection


class Status(StrEnum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"  # no sufficient evidence in the passages, not proof of absence
    UNCERTAIN = "UNCERTAIN"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Normalized values (interpretation; never used as evidence) ---------------------------


class GoverningLawValue(_Strict):
    jurisdiction: str | None = Field(
        description="State or country whose law governs, e.g. 'New York'; null if unclear."
    )


class TerminationForConvenienceValue(_Strict):
    party_with_right: str | None = Field(
        description="Who may terminate without cause, as named in the contract; null if unclear."
    )
    notice_days: int | None = Field(
        description="Required notice in days if stated as a number of days; otherwise null."
    )


class CapOnLiabilityValue(_Strict):
    has_cap: bool | None = Field(
        description="True if liability is capped in amount or in time to bring claims; "
        "null if unclear."
    )
    cap_expression: str | None = Field(
        description="The cap amount, formula or time limit in short form; null if none or unclear."
    )
    exceptions: list[str] = Field(description="Carve-outs from the cap, briefly; empty if none.")


class RenewalTermValue(_Strict):
    automatic_renewal: bool | None = Field(
        description="True only if automatic renewal is explicitly established; null if unclear."
    )
    renewal_term: str | None = Field(
        description="Length of each renewal term, e.g. 'successive 1-year terms'; null if unclear."
    )


class RenewalNoticeValue(_Strict):
    notice_days: int | None = Field(
        description="Notice in days required to prevent renewal, if stated in days; else null."
    )


NormalizedValue = (
    GoverningLawValue
    | TerminationForConvenienceValue
    | CapOnLiabilityValue
    | RenewalTermValue
    | RenewalNoticeValue
)

VALUE_MODELS: dict[ClauseType, type[NormalizedValue]] = {
    ClauseType.GOVERNING_LAW: GoverningLawValue,
    ClauseType.TERMINATION_FOR_CONVENIENCE: TerminationForConvenienceValue,
    ClauseType.CAP_ON_LIABILITY: CapOnLiabilityValue,
    ClauseType.RENEWAL_TERM: RenewalTermValue,
    ClauseType.RENEWAL_NOTICE_PERIOD: RenewalNoticeValue,
}

# --- Model answers ------------------------------------------------------------------------


class Quote(_Strict):
    passage_id: str = Field(description="ID of the passage the quote is copied from, e.g. 'P2'.")
    text: str = Field(description="Exact contiguous text copied verbatim from that passage.")


class ClauseAnswer(_Strict):
    status: Status = Field(
        description="FOUND if the passages establish the clause; NOT_FOUND if they do not; "
        "UNCERTAIN if they are ambiguous or incomplete."
    )
    quotes: list[Quote] = Field(
        description="Verbatim quotes supporting the answer, most relevant first; "
        "empty if NOT_FOUND."
    )
    notes: str = Field(description="Brief reasoning or caveats; empty string if none.")


class GoverningLawAnswer(ClauseAnswer):
    value: GoverningLawValue


class TerminationForConvenienceAnswer(ClauseAnswer):
    value: TerminationForConvenienceValue


class CapOnLiabilityAnswer(ClauseAnswer):
    value: CapOnLiabilityValue


class RenewalTermAnswer(ClauseAnswer):
    value: RenewalTermValue


class RenewalNoticeAnswer(ClauseAnswer):
    value: RenewalNoticeValue


ANSWER_MODELS: dict[ClauseType, type[ClauseAnswer]] = {
    ClauseType.GOVERNING_LAW: GoverningLawAnswer,
    ClauseType.TERMINATION_FOR_CONVENIENCE: TerminationForConvenienceAnswer,
    ClauseType.CAP_ON_LIABILITY: CapOnLiabilityAnswer,
    ClauseType.RENEWAL_TERM: RenewalTermAnswer,
    ClauseType.RENEWAL_NOTICE_PERIOD: RenewalNoticeAnswer,
}

# --- Verified results ---------------------------------------------------------------------


class Evidence(_Strict):
    """A quote verified to exist in the contract at [source_start, source_end)."""

    document_id: str
    chunk_id: str
    source_start: int = Field(ge=0)
    source_end: int
    exact_quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.source_end != self.source_start + len(self.exact_quote):
            raise ValueError("source_end must equal source_start + len(exact_quote)")
        return self


class LLMUsage(_Strict):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class ClauseResult(_Strict):
    clause_type: ClauseType
    status: Status
    extracted_text: str | None
    normalized_value: NormalizedValue | None
    evidence: list[Evidence]
    extraction_notes: list[str]
    retrieved_chunk_ids: list[str]
    usage: LLMUsage | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_value_by_clause_type(cls, data: Any) -> Any:
        """Union members share field names; pick the model from clause_type, not by guessing."""
        if isinstance(data, dict) and isinstance(data.get("normalized_value"), dict):
            model = VALUE_MODELS[ClauseType(data["clause_type"])]
            data = {**data, "normalized_value": model.model_validate(data["normalized_value"])}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        value = self.normalized_value
        if value is not None and type(value) is not VALUE_MODELS[self.clause_type]:
            raise ValueError(f"normalized_value type does not match {self.clause_type}")
        if self.status == Status.FOUND and not self.evidence:
            raise ValueError("FOUND requires verified evidence")
        if self.status == Status.NOT_FOUND and (self.evidence or self.extracted_text or value):
            raise ValueError("NOT_FOUND carries no evidence, text or value")
        expected = self.evidence[0].exact_quote if self.evidence else None
        if self.extracted_text != expected:
            raise ValueError("extracted_text must be the first verified evidence quote")
        return self

    @property
    def section(self) -> ReportSection:
        return SECTION_OF[self.clause_type]
