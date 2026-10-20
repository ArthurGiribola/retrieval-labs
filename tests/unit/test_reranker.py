from fake_models import HashingEmbedder, OverlapScorer
from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import to_result
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.dense import DenseRetriever
from retrieval_labs.retrieval.hybrid import HybridRetriever
from retrieval_labs.retrieval.reranker import MINILM_RERANKER, RerankingRetriever


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


TEXTS = [
    "liability liability",
    "limitation of liability shall not exceed fees paid",
    "unrelated payment schedule",
]
QUERY = "limitation of liability shall not exceed fees"


class FixedOrder:
    """Base retriever with a known order, so reranking is tested in isolation."""

    name = "fixed"

    def __init__(self, order: list[int]) -> None:
        self.order = order

    def build(self, chunks):
        order = self.order

        class Index:
            def search(self, query: str, top_k: int):
                return [
                    to_result(chunks[i], rank, 1.0 / rank)
                    for rank, i in enumerate(order[:top_k], start=1)
                ]

        return Index()


def test_reranker_reorders_base_candidates() -> None:
    chunks = make_chunks(TEXTS)
    base = FixedOrder([0, 2, 1])
    reranked = RerankingRetriever(base, OverlapScorer()).build(chunks).search(QUERY, 3)
    # Overlap scores: chunk 1 has 7 distinct query terms, chunk 0 has 1, chunk 2 has 0.
    assert [r.chunk_id for r in reranked] == ["c1#0001", "c1#0000", "c1#0002"]
    assert [r.score for r in reranked] == [7.0, 1.0, 0.0]
    assert [r.rank for r in reranked] == [1, 2, 3]


def test_only_top_n_candidates_are_scored() -> None:
    chunks = make_chunks([f"term{i} liability" for i in range(30)])
    scorer = OverlapScorer()
    results = (
        RerankingRetriever(BM25Retriever(), scorer, candidates=20)
        .build(chunks)
        .search("liability", top_k=50)
    )
    assert scorer.pairs_scored == 20
    assert len(results) == 20


def test_ties_keep_base_order() -> None:
    chunks = make_chunks(["fees paid", "fees paid here", "fees paid there"])
    base = BM25Retriever().build(chunks).search("fees paid", 3)
    reranked = (
        RerankingRetriever(BM25Retriever(), OverlapScorer()).build(chunks).search("fees paid", 3)
    )
    assert [r.chunk_id for r in reranked] == [r.chunk_id for r in base]


def test_offsets_and_text_are_preserved() -> None:
    chunks = make_chunks(TEXTS)
    by_id = {c.chunk_id: c for c in chunks}
    for result in (
        RerankingRetriever(BM25Retriever(), OverlapScorer()).build(chunks).search(QUERY, 3)
    ):
        chunk = by_id[result.chunk_id]
        assert (result.start, result.end, result.text) == (chunk.start, chunk.end, chunk.text)


def test_name_and_params_describe_the_pipeline() -> None:
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEmbedder())])
    retriever = RerankingRetriever(hybrid, OverlapScorer())
    assert retriever.name == "hybrid+rerank"
    assert retriever.params.model_dump() == {
        "base": "hybrid",
        "model": "fake-overlap",
        "candidates": 20,
    }


def test_reranker_spec_is_pinned() -> None:
    assert MINILM_RERANKER.revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert MINILM_RERANKER.repo == "cross-encoder/ms-marco-MiniLM-L6-v2"
