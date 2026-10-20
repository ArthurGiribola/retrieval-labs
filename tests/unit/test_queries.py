from retrieval_labs.clauses import ClauseType
from retrieval_labs.documents import Contract
from retrieval_labs.ingestion.chunking import ChunkingConfig
from retrieval_labs.pipeline import PipelineConfig
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.contract_search import ContractSearch
from retrieval_labs.retrieval.queries import EXPANDED_QUERIES, QUERIES

EXPANDED_TYPES = {ClauseType.CAP_ON_LIABILITY, ClauseType.TERMINATION_FOR_CONVENIENCE}


def test_expansion_only_touches_the_two_failing_clause_types() -> None:
    changed = {ct for ct in ClauseType if EXPANDED_QUERIES[ct] != QUERIES[ct]}
    assert changed == EXPANDED_TYPES


def test_expanded_queries_extend_the_frozen_ones() -> None:
    for clause_type in EXPANDED_TYPES:
        assert EXPANDED_QUERIES[clause_type].startswith(QUERIES[clause_type] + " ")


def test_contract_search_uses_frozen_queries_by_default() -> None:
    text = "In no event shall either party be liable for consequential damages."
    contract = Contract.from_text("c1", "t", text)
    config = ChunkingConfig(window_words=20, overlap_words=5)
    frozen = ContractSearch(contract, BM25Retriever(), config)
    expanded = ContractSearch(contract, BM25Retriever(), config, EXPANDED_QUERIES)
    cap = ClauseType.CAP_ON_LIABILITY
    assert frozen.search(cap, 1)[0].score < expanded.search(cap, 1)[0].score


def test_the_application_pipeline_never_uses_the_expanded_queries() -> None:
    import inspect

    import retrieval_labs.pipeline as pipeline

    assert "EXPANDED_QUERIES" not in inspect.getsource(pipeline)
    assert PipelineConfig().top_k == 5
