"""Gold annotations. EVALUATION ONLY: inference code must never import this module.

The import boundary is enforced by ``tests/unit/test_leakage.py``.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.documents import Contract


class GoldSpan(BaseModel):
    """One annotated answer span; ``end`` is exclusive, offsets index the original text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_length(self) -> Self:
        if self.end != self.start + len(self.text):
            raise ValueError(f"span end {self.end} != start {self.start} + len(text)")
        return self


class GoldAnnotation(BaseModel):
    """All gold spans for one (contract, label) pair. No spans means the label is absent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str = Field(min_length=1)
    label: CuadLabel
    spans: tuple[GoldSpan, ...]

    @model_validator(mode="after")
    def _check_spans_sorted_and_unique(self) -> Self:
        keys = [(s.start, s.end) for s in self.spans]
        if keys != sorted(set(keys)):
            raise ValueError("spans must be sorted by (start, end) and unique")
        return self

    @property
    def is_answerable(self) -> bool:
        return bool(self.spans)


def check_offsets(annotation: GoldAnnotation, contract: Contract) -> None:
    """Raise ``CuadDataError`` unless every span matches the contract text at its offsets."""
    if annotation.contract_id != contract.contract_id:
        raise CuadDataError(
            f"annotation for {annotation.contract_id!r} checked against {contract.contract_id!r}"
        )
    for span in annotation.spans:
        if contract.text[span.start : span.end] != span.text:
            raise CuadDataError(
                f"offset mismatch in {contract.contract_id!r} / {annotation.label!s} "
                f"at [{span.start}, {span.end})"
            )
