"""Chunk, index and query one contract: the retrieval path shared by the application
pipeline and the benchmark, so both always measure the same algorithm."""

from __future__ import annotations

import time
from collections.abc import Mapping

from retrieval_labs.clauses import ClauseType
from retrieval_labs.documents import Contract
from retrieval_labs.ingestion.chunking import ChunkingConfig, chunk_contract
from retrieval_labs.retrieval.base import RetrievalResult, Retriever
from retrieval_labs.retrieval.queries import QUERIES


class ContractSearch:
    def __init__(
        self,
        contract: Contract,
        retriever: Retriever,
        chunking: ChunkingConfig,
        queries: Mapping[ClauseType, str] = QUERIES,
    ) -> None:
        self._queries = queries
        started = time.perf_counter()
        self.chunks = chunk_contract(contract, chunking)
        chunked = time.perf_counter()
        self._index = retriever.build(self.chunks) if self.chunks else None
        self.timings_s = {
            "chunking": chunked - started,
            "indexing": time.perf_counter() - chunked,
        }

    def search(self, clause_type: ClauseType, top_k: int) -> list[RetrievalResult]:
        if self._index is None:
            return []
        return self._index.search(self._queries[clause_type], top_k)
