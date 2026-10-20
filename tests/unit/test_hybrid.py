from collections.abc import Sequence

import pytest

from fake_models import HashingEmbedder
from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import RetrievalResult, to_result
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.dense import DenseRetriever
from retrieval_labs.retrieval.hybrid import HybridRetriever


def make_chunks(texts: list[str]) -> list[Chunk]:
    chunks, offset = [], 0
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"c1#{index:04d}",
                contract_id="c1",
                index=index,
                start=offset,
                end=offset + len(text),
                text=text,
                chunking="test",
            )
        )
        offset += len(text) + 1
    return chunks


CHUNKS = make_chunks(["alpha", "beta", "gamma", "delta"])
A, B, C, D = CHUNKS


class FixedIndex:
    def __init__(self, order: list[Chunk]) -> None:
        self.order = order

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [to_result(c, rank, 1.0) for rank, c in enumerate(self.order[:top_k], start=1)]


class FixedRetriever:
    def __init__(self, name: str, order: list[Chunk]) -> None:
        self.name = name
        self.order = order

    def build(self, chunks: Sequence[Chunk]) -> FixedIndex:
        return FixedIndex(self.order)


def test_rrf_scores_match_hand_computation() -> None:
    lexical = FixedRetriever("lexical", [A, B])  # C and D absent, as BM25 omits zero scores
    semantic = FixedRetriever("semantic", [B, C, A])
    results = HybridRetriever([lexical, semantic], rrf_k=60).build(CHUNKS).search("q", top_k=4)
    expected = {
        B.chunk_id: 1 / 62 + 1 / 61,
        A.chunk_id: 1 / 61 + 1 / 63,
        C.chunk_id: 1 / 62,
    }
    assert [r.chunk_id for r in results] == [B.chunk_id, A.chunk_id, C.chunk_id]
    for result in results:
        assert result.score == pytest.approx(expected[result.chunk_id])
    assert [r.rank for r in results] == [1, 2, 3]


def test_ties_are_broken_by_document_order() -> None:
    first = FixedRetriever("one", [C, A])
    second = FixedRetriever("two", [A, C])
    results = HybridRetriever([first, second]).build(CHUNKS).search("q", top_k=2)
    assert [r.chunk_id for r in results] == [A.chunk_id, C.chunk_id]  # equal fused scores


def test_top_k_limits_results() -> None:
    hybrid = HybridRetriever([FixedRetriever("x", [A, B, C, D])])
    assert len(hybrid.build(CHUNKS).search("q", top_k=2)) == 2


def test_params_record_components_and_k() -> None:
    params = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEmbedder())]).params
    assert params.components == ("bm25", "dense")
    assert params.rrf_k == 60


def test_hybrid_over_real_components_returns_offsets() -> None:
    texts = ["payment within thirty days", "governed by the laws of new york", "delivery terms"]
    chunks = make_chunks(texts)
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEmbedder())])
    results = hybrid.build(chunks).search("laws of new york", top_k=3)
    assert results[0].chunk_id == "c1#0001"
    assert {r.chunk_id for r in results} == {c.chunk_id for c in chunks}
