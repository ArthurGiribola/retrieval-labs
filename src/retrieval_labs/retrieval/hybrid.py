"""Hybrid retrieval by Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009).

    fused(chunk) = sum over component rankings r of 1 / (k + rank_r(chunk))

Only ranks are fused, so BM25 scores and cosine similarities never need to be put on a
common scale. A chunk missing from a component's ranking (BM25 omits chunks that share no
term with the query) simply gets no contribution from it. k = 60 is the value from the
original paper, fixed a priori rather than tuned on dev.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import (
    RetrievalResult,
    Retriever,
    SearchIndex,
    single_contract_id,
    to_result,
)

RRF_K = 60


class HybridParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rrf_k: int = Field(default=RRF_K, gt=0)
    components: tuple[str, ...]


class HybridIndex:
    def __init__(self, chunks: Sequence[Chunk], indexes: Sequence[SearchIndex], rrf_k: int) -> None:
        single_contract_id(chunks)
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._indexes = list(indexes)
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        fused: dict[str, float] = defaultdict(float)
        for index in self._indexes:  # fixed component order keeps float sums reproducible
            for result in index.search(query, len(self._chunks)):
                fused[result.chunk_id] += 1.0 / (self._rrf_k + result.rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], self._chunks[item[0]].index))
        return [
            to_result(self._chunks[chunk_id], rank, score)
            for rank, (chunk_id, score) in enumerate(ranked[:top_k], start=1)
        ]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = RRF_K) -> None:
        self.retrievers = list(retrievers)
        self.params = HybridParams(rrf_k=rrf_k, components=tuple(r.name for r in retrievers))

    def build(self, chunks: Sequence[Chunk]) -> HybridIndex:
        return HybridIndex(chunks, [r.build(chunks) for r in self.retrievers], self.params.rrf_k)
