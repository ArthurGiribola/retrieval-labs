import math

import pytest

from retrieval_labs.clauses import ClauseType
from retrieval_labs.documents import Contract
from retrieval_labs.ingestion.chunking import Chunk, ChunkingConfig, chunk_contract
from retrieval_labs.retrieval.bm25 import BM25Params, BM25Retriever, tokenize
from retrieval_labs.retrieval.queries import QUERIES


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


DOCS = ["Governing law: New York", "Law firm", "Payment terms net thirty"]


def test_tokenize() -> None:
    assert tokenize("Governed by the LAWS of New-York, U.S.A. (2020)") == [
        "governed", "by", "the", "laws", "of", "new", "york", "u", "s", "a", "2020",
    ]  # fmt: skip


def test_scores_match_hand_computed_values() -> None:
    """Expected values written out from the formula, independent of the implementation."""
    index = BM25Retriever(BM25Params(k1=1.2, b=0.75)).build(make_chunks(DOCS))
    results = index.search("governing law", top_k=3)
    avgdl = (4 + 2 + 4) / 3
    idf_governing = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    idf_law = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_doc0 = (idf_governing + idf_law) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 4 / avgdl))
    expected_doc1 = idf_law * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / avgdl))
    assert [r.chunk_id for r in results] == ["c1#0000", "c1#0001"]
    assert results[0].score == pytest.approx(expected_doc0)
    assert results[1].score == pytest.approx(expected_doc1)
    assert [r.rank for r in results] == [1, 2]


def test_idf_stays_positive_for_terms_in_every_chunk() -> None:
    index = BM25Retriever().build(make_chunks(["the agreement", "the agreement", "the party"]))
    assert index.idf("the") > 0
    assert index.idf("agreement") > 0
    assert index.idf("the") < index.idf("agreement") < index.idf("party")


def test_repeated_query_terms_count_once() -> None:
    index = BM25Retriever().build(make_chunks(DOCS))
    once = index.search("law", top_k=1)[0].score
    twice = index.search("law law LAW", top_k=1)[0].score
    assert once == twice


def test_chunks_without_query_terms_are_not_returned() -> None:
    index = BM25Retriever().build(make_chunks(DOCS))
    assert [r.chunk_id for r in index.search("payment", top_k=5)] == ["c1#0002"]
    assert index.search("arbitration", top_k=5) == []


def test_ties_are_broken_by_document_order() -> None:
    index = BM25Retriever().build(make_chunks(["notice here", "other text", "notice here"]))
    assert [r.chunk_id for r in index.search("notice", top_k=2)] == ["c1#0000", "c1#0002"]


def test_top_k_limits_results() -> None:
    index = BM25Retriever().build(make_chunks(["a law", "b law", "c law", "d law"]))
    assert len(index.search("law", top_k=2)) == 2


def test_index_rejects_chunks_from_several_contracts() -> None:
    chunks = make_chunks(["a"], "c1") + make_chunks(["b"], "c2")
    with pytest.raises(ValueError, match="exactly one contract"):
        BM25Retriever().build(chunks)


def test_results_carry_source_offsets() -> None:
    text = "Preamble text here. This Agreement is governed by the laws of Delaware. End."
    contract = Contract.from_text("c9", "t", text)
    chunks = chunk_contract(contract, ChunkingConfig(window_words=4, overlap_words=1))
    for result in BM25Retriever().build(chunks).search(QUERIES[ClauseType.GOVERNING_LAW], 3):
        assert text[result.start : result.end] == result.text
        assert result.contract_id == "c9"


def test_every_clause_type_has_a_query() -> None:
    assert set(QUERIES) == set(ClauseType)
