import pytest

from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.factory import (
    APPLICATION_METHOD,
    DENSE_BASED,
    METHODS,
    build_retriever,
    build_retrievers,
)


def test_application_method_is_a_measured_method() -> None:
    assert APPLICATION_METHOD == "hybrid+rerank"
    assert APPLICATION_METHOD in METHODS
    assert set(DENSE_BASED) <= set(METHODS)


def test_bm25_builds_without_loading_models() -> None:
    retrievers, info = build_retrievers(["bm25"])
    assert list(retrievers) == ["bm25"]
    assert isinstance(retrievers["bm25"], BM25Retriever)
    assert info == {}
    assert isinstance(build_retriever("bm25"), BM25Retriever)


def test_unknown_method_is_rejected_before_loading_anything() -> None:
    with pytest.raises(ValueError, match="bm25-expanded"):
        build_retrievers(["bm25", "bm25-expanded"])
