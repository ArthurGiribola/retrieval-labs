from pathlib import Path

import numpy as np
import pytest

from fake_models import HashingEmbedder
from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.dense import DenseRetriever
from retrieval_labs.retrieval.embeddings import BGE_SMALL, CachedEmbedder


def make_chunks(texts: list[str], contract_id: str = "c1") -> list[Chunk]:
    chunks, offset = [], 0
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{contract_id}#{index:04d}",
                contract_id=contract_id,
                index=index,
                start=offset,
                end=offset + len(text),
                text=text,
                chunking="test",
            )
        )
        offset += len(text) + 1
    return chunks


DOCS = [
    "payment terms net thirty days",
    "this agreement is governed by the laws of new york",
    "the supplier shall deliver products",
]


def test_ranks_by_cosine_similarity() -> None:
    index = DenseRetriever(HashingEmbedder()).build(make_chunks(DOCS))
    results = index.search("governed by the laws of the state", top_k=3)
    assert results[0].chunk_id == "c1#0001"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(max(r.score for r in results))
    assert all(-1.0 - 1e-6 <= r.score <= 1.0 + 1e-6 for r in results)


def test_every_chunk_gets_a_score_unlike_bm25() -> None:
    index = DenseRetriever(HashingEmbedder()).build(make_chunks(DOCS))
    assert len(index.search("arbitration in london", top_k=10)) == 3


def test_ties_are_broken_by_document_order() -> None:
    index = DenseRetriever(HashingEmbedder()).build(
        make_chunks(["same text", "other", "same text"])
    )
    assert [r.chunk_id for r in index.search("same text", top_k=2)] == ["c1#0000", "c1#0002"]


def test_index_rejects_chunks_from_several_contracts() -> None:
    chunks = make_chunks(["a"], "c1") + make_chunks(["b"], "c2")
    with pytest.raises(ValueError, match="exactly one contract"):
        DenseRetriever(HashingEmbedder()).build(chunks)


def test_params_record_the_model() -> None:
    assert DenseRetriever(HashingEmbedder(name="m@1")).params.model == "m@1"


def test_bge_spec_is_pinned_with_documented_query_prefix() -> None:
    assert BGE_SMALL.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert BGE_SMALL.query_prefix.startswith("Represent this sentence for searching")
    assert BGE_SMALL.name == "BAAI/bge-small-en-v1.5@5c38ec7c"


# --- cache ---


def test_cache_reuses_vectors_across_instances(tmp_path: Path) -> None:
    inner = HashingEmbedder()
    first = CachedEmbedder(inner, tmp_path).embed_documents(DOCS)
    second_cache = CachedEmbedder(inner, tmp_path)
    second = second_cache.embed_documents(DOCS)
    assert inner.document_calls == 1
    assert (second_cache.hits, second_cache.misses) == (1, 0)
    np.testing.assert_array_equal(first, second)


def test_cache_key_depends_on_exact_texts_and_model(tmp_path: Path) -> None:
    inner = HashingEmbedder()
    cache = CachedEmbedder(inner, tmp_path)
    cache.embed_documents(DOCS)
    cache.embed_documents([*DOCS[:2], DOCS[2] + " "])  # one character differs
    CachedEmbedder(HashingEmbedder(name="other"), tmp_path).embed_documents(DOCS)
    assert cache.misses == 2
    assert len(list(tmp_path.rglob("*.npy"))) == 3


def test_cache_key_is_not_fooled_by_text_boundaries(tmp_path: Path) -> None:
    cache = CachedEmbedder(HashingEmbedder(), tmp_path)
    cache.embed_documents(["ab", "c"])
    cache.embed_documents(["a", "bc"])
    assert cache.misses == 2


def test_cached_dense_retrieval_matches_uncached(tmp_path: Path) -> None:
    chunks = make_chunks(DOCS)
    plain = DenseRetriever(HashingEmbedder()).build(chunks).search("laws of new york", 3)
    cached_retriever = DenseRetriever(CachedEmbedder(HashingEmbedder(), tmp_path))
    cached_retriever.build(chunks)
    cached = cached_retriever.build(chunks).search("laws of new york", 3)
    assert plain == cached
