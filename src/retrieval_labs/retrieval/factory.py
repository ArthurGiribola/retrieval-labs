"""Build retrieval methods by name: shared by the benchmark, the pipeline and the demo, so the
application runs exactly the configuration that was measured."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.bm25 import BM25Retriever

METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")
DENSE_BASED = ("dense", "hybrid", "hybrid+rerank")
# Owner decision after Phase 4 (docs/DECISIONS.md): best fair method and best presence signal.
APPLICATION_METHOD = "hybrid+rerank"
EMBEDDING_CACHE_DIR = Path("data/cache/embeddings")


def build_retrievers(
    methods: list[str], cache_dir: Path = EMBEDDING_CACHE_DIR
) -> tuple[dict[str, Retriever], dict[str, Any]]:
    """Instantiate the requested methods, loading each model once (model imports are lazy).

    The returned info holds model load times and, when a dense method is requested, the
    ``CachedEmbedder`` under ``"embedder"`` so callers can read its cache hits and misses.
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown retrieval method(s): {unknown}; choose from {METHODS}")
    retrievers: dict[str, Retriever] = {"bm25": BM25Retriever()}
    info: dict[str, Any] = {}
    if any(m in DENSE_BASED for m in methods):
        from retrieval_labs.retrieval.dense import DenseRetriever
        from retrieval_labs.retrieval.embeddings import CachedEmbedder, SentenceTransformerEmbedder
        from retrieval_labs.retrieval.hybrid import HybridRetriever

        started = time.perf_counter()
        model = SentenceTransformerEmbedder()
        info["embedder_load_s"] = round(time.perf_counter() - started, 3)
        info["embedder_max_seq_length"] = model.max_seq_length
        embedder = CachedEmbedder(model, cache_dir, memoize_queries=False)
        info["embedder"] = embedder
        retrievers["dense"] = DenseRetriever(embedder)
        retrievers["hybrid"] = HybridRetriever([retrievers["bm25"], retrievers["dense"]])
    if "hybrid+rerank" in methods:
        from retrieval_labs.retrieval.reranker import (
            RerankingRetriever,
            SentenceTransformerCrossScorer,
        )

        started = time.perf_counter()
        scorer = SentenceTransformerCrossScorer()
        info["reranker_load_s"] = round(time.perf_counter() - started, 3)
        retrievers["hybrid+rerank"] = RerankingRetriever(retrievers["hybrid"], scorer)
    return {m: retrievers[m] for m in methods}, info


def build_retriever(method: str = APPLICATION_METHOD) -> Retriever:
    return build_retrievers([method])[0][method]
