"""Split a contract into overlapping word windows that keep exact source offsets.

The text is never normalized: each chunk's text is the slice
``contract.text[start:end]``, running from the first character of its first word
to the last character of its last word. Words are maximal runs of non-whitespace.

Coverage guarantee: with a stride of ``window - overlap`` words, any run of at most
``overlap + 1`` consecutive words lies entirely inside at least one chunk.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval_labs.documents import Contract

_WORD_RE = re.compile(r"\S+")


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    window_words: int = Field(default=150, gt=0)
    overlap_words: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.overlap_words >= self.window_words:
            raise ValueError("overlap_words must be smaller than window_words")
        return self

    @property
    def stride(self) -> int:
        return self.window_words - self.overlap_words

    @property
    def label(self) -> str:
        return f"words{self.window_words}-overlap{self.overlap_words}"


class Chunk(BaseModel):
    """A retrievable passage. ``end`` is exclusive; offsets index the original text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    contract_id: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int
    text: str = Field(min_length=1)
    chunking: str

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end != self.start + len(self.text):
            raise ValueError("chunk end must equal start + len(text)")
        return self


def chunk_contract(contract: Contract, config: ChunkingConfig) -> list[Chunk]:
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(contract.text)]
    chunks: list[Chunk] = []
    for index, first in enumerate(range(0, len(words), config.stride)):
        last = min(first + config.window_words, len(words)) - 1
        start, end = words[first][0], words[last][1]
        chunks.append(
            Chunk(
                chunk_id=f"{contract.contract_id}#{index:04d}",
                contract_id=contract.contract_id,
                index=index,
                start=start,
                end=end,
                text=contract.text[start:end],
                chunking=config.label,
            )
        )
        if last == len(words) - 1:
            break
    return chunks
