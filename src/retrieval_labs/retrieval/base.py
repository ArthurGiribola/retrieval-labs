"""The interface every retrieval method implements (BM25 now; dense and hybrid later).

An index is built from the chunks of exactly one contract, so retrieval can never
return passages from another document.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from retrieval_labs.ingestion.chunking import Chunk


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    contract_id: str
    rank: int = Field(ge=1)
    score: float
    start: int
    end: int
    text: str


class SearchIndex(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievalResult]: ...


class Retriever(Protocol):
    name: str

    def build(self, chunks: Sequence[Chunk]) -> SearchIndex: ...


def single_contract_id(chunks: Sequence[Chunk]) -> str:
    """The contract all chunks belong to; raises if they span several contracts."""
    ids = {chunk.contract_id for chunk in chunks}
    if len(ids) != 1:
        raise ValueError(f"an index must cover exactly one contract, got {len(ids)}")
    return ids.pop()


def to_result(chunk: Chunk, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        contract_id=chunk.contract_id,
        rank=rank,
        score=score,
        start=chunk.start,
        end=chunk.end,
        text=chunk.text,
    )
