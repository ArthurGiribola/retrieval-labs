"""Source documents: the only contract representation that inference code may receive."""

from __future__ import annotations

import hashlib
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Contract(BaseModel):
    """A source contract: identifier, title and the exact original text.

    It deliberately carries no annotation data. Gold labels live in
    ``retrieval_labs.cuad.gold`` and must never be attached to this type.
    ``text_sha256`` lets any consumer verify the text was not altered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_sha256: str

    @classmethod
    def from_text(cls, contract_id: str, title: str, text: str) -> Contract:
        return cls(contract_id=contract_id, title=title, text=text, text_sha256=sha256_text(text))

    @model_validator(mode="after")
    def _check_text_hash(self) -> Self:
        if sha256_text(self.text) != self.text_sha256:
            raise ValueError(f"text_sha256 does not match text for {self.contract_id!r}")
        return self
