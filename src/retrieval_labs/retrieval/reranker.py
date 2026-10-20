"""Second-stage reranking: a cross-encoder rescores a base method's top candidates.

The reranker sees a deeper candidate list (20) than the final context budget (5); this
difference from the other methods is documented in the benchmark (brief Section 9.6).
Like ``embeddings.py``, this module imports sentence-transformers lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import RetrievalResult, Retriever, SearchIndex
from retrieval_labs.retrieval.embeddings import ModelSpec

RERANK_CANDIDATES = 20

# Verified 2026-10-20 on the Hugging Face API: Apache-2.0, 22.7M parameters, 512-token input.
# Trained on MS MARCO web queries; our queries are keyword lists (a measured, not assumed, fit).
MINILM_RERANKER = ModelSpec(
    repo="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
)


class CrossScorer(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class SentenceTransformerCrossScorer:
    def __init__(self, spec: ModelSpec = MINILM_RERANKER, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.name = spec.name
        self.batch_size = batch_size
        self._model = CrossEncoder(spec.repo, revision=spec.revision, device="cpu")

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


class RerankParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base: str
    model: str
    candidates: int = Field(default=RERANK_CANDIDATES, gt=0)


class RerankingIndex:
    def __init__(self, base: SearchIndex, scorer: CrossScorer, candidates: int) -> None:
        self._base = base
        self._scorer = scorer
        self._candidates = candidates

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Rerank the base top-N; returns at most N results whatever top_k is."""
        candidates = self._base.search(query, self._candidates)
        scores = self._scorer.score(query, [c.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
        return [
            candidates[i].model_copy(update={"rank": rank, "score": scores[i]})
            for rank, i in enumerate(order[:top_k], start=1)
        ]


class RerankingRetriever:
    def __init__(
        self, base: Retriever, scorer: CrossScorer, candidates: int = RERANK_CANDIDATES
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.name = f"{base.name}+rerank"
        self.params = RerankParams(base=base.name, model=scorer.name, candidates=candidates)

    def build(self, chunks: Sequence[Chunk]) -> RerankingIndex:
        return RerankingIndex(self.base.build(chunks), self.scorer, self.params.candidates)
