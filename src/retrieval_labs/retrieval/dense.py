"""Dense retrieval: cosine similarity between a query embedding and chunk embeddings.

Vectors are L2-normalized, so the dot product is the cosine. Every chunk gets a score
(unlike BM25, which never returns chunks without a shared term); ties are broken by
document order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import RetrievalResult, single_contract_id, to_result
from retrieval_labs.retrieval.embeddings import Embedder


class DenseParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        single_contract_id(chunks)
        self._chunks = list(chunks)
        self._embedder = embedder
        self._vectors = embedder.embed_documents([chunk.text for chunk in self._chunks])

    def scores(self, query: str) -> np.ndarray:
        return self._vectors @ self._embedder.embed_query(query)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        scores = self.scores(query)
        positions = np.arange(len(scores))
        order = np.lexsort((positions, -scores))[:top_k]  # score desc, then document order
        return [
            to_result(self._chunks[i], rank, float(scores[i]))
            for rank, i in enumerate(order.tolist(), start=1)
        ]


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.params = DenseParams(model=embedder.name)

    def build(self, chunks: Sequence[Chunk]) -> DenseIndex:
        return DenseIndex(chunks, self.embedder)
