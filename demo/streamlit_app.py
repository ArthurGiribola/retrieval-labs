"""Retrieval Labs demo: the evidence retrieved for each clause type in a CUAD contract, and the
benchmark results behind the choice of retriever. Makes no paid API calls.

Run from the repository root, after preparing the dataset (scripts/prepare_cuad.py --download):

    uv run streamlit run demo/streamlit_app.py

Optional environment variables:
    RETRIEVAL_LABS_DEMO_RETRIEVER  retrieval method (default: hybrid+rerank; "bm25" loads no models)
    RETRIEVAL_LABS_PROCESSED_DIR   prepared dataset directory (default: data/processed/cuad)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from retrieval_labs.clauses import SECTION_OF, ClauseType, ReportSection
from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import load_contracts
from retrieval_labs.documents import Contract
from retrieval_labs.evaluation.report_tables import (
    DEV_RUNS,
    TEST_RUN,
    load_rows,
    markdown_table,
    presence_thresholds,
)
from retrieval_labs.ingestion.chunking import ChunkingConfig
from retrieval_labs.retrieval.base import RetrievalResult, Retriever
from retrieval_labs.retrieval.contract_search import ContractSearch
from retrieval_labs.retrieval.factory import APPLICATION_METHOD, build_retrievers

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports" / "benchmark"
CACHE_DIR = ROOT / "data" / "cache" / "embeddings"
PROCESSED_DIR = Path(
    os.environ.get("RETRIEVAL_LABS_PROCESSED_DIR", ROOT / "data" / "processed" / "cuad")
)
METHOD = os.environ.get("RETRIEVAL_LABS_DEMO_RETRIEVER", APPLICATION_METHOD)
TOP_K = 5

CLAUSE_NAMES = {
    ClauseType.GOVERNING_LAW: "Governing law",
    ClauseType.TERMINATION_FOR_CONVENIENCE: "Termination for convenience",
    ClauseType.CAP_ON_LIABILITY: "Cap on liability",
    ClauseType.RENEWAL_TERM: "Renewal term",
    ClauseType.RENEWAL_NOTICE_PERIOD: "Notice period to terminate renewal",
}


@st.cache_resource(show_spinner="Loading retrieval models (first run only)...")
def load_retriever(method: str) -> Retriever:
    return build_retrievers([method], CACHE_DIR)[0][method]


@st.cache_resource(show_spinner="Loading contracts...")
def load_test_contracts(directory: Path) -> dict[str, Contract]:
    contracts = load_contracts(Split.TEST, directory)
    return {c.contract_id: c for c in sorted(contracts, key=lambda c: c.title.lower())}


@st.cache_data(show_spinner=False, max_entries=32)
def retrieve(
    contract_id: str, method: str, directory: Path
) -> tuple[dict[ClauseType, list[RetrievalResult]], float]:
    """Top passages per clause type for one contract, and the seconds retrieval took
    (indexing, including chunk embedding when not cached, plus the five queries)."""
    contract, retriever = load_test_contracts(directory)[contract_id], load_retriever(method)
    started = time.perf_counter()
    search = ContractSearch(contract, retriever, ChunkingConfig())
    results = {clause: search.search(clause, TOP_K) for clause in ClauseType}
    return results, time.perf_counter() - started


def show_passage(result: RetrievalResult) -> None:
    st.caption(
        f"Rank {result.rank} · score {result.score:.3f} · "
        f"characters {result.start:,}–{result.end:,}"
    )
    st.code(result.text, language=None, wrap_lines=True)


def show_clause(clause: ClauseType, results: list[RetrievalResult], threshold: float) -> None:
    top = results[0] if results else None
    above = top is not None and top.score >= threshold
    st.markdown(f"**{CLAUSE_NAMES[clause]}**")
    if above:
        st.badge("Evidence above presence threshold", color="green")
    else:
        st.badge("Insufficient evidence", color="gray")
    score = f"{top.score:.3f}" if top else "none"
    st.caption(f"Top-1 score {score}; threshold {threshold:.3f} (fit on the train split)")
    if top is None:
        return
    show_passage(top)
    with st.expander(f"Passages 2–{len(results)}"):
        for result in results[1:]:
            show_passage(result)


def evidence_tab() -> None:
    try:
        contracts = load_test_contracts(PROCESSED_DIR)
    except CuadDataError as exc:
        st.error(
            f"The prepared CUAD dataset was not found ({exc}). Run "
            "`uv run python scripts/prepare_cuad.py --download` from the repository root."
        )
        return
    contract_id = st.selectbox(
        f"Contract ({len(contracts)} contracts of the CUAD test split)",
        options=list(contracts),
        format_func=lambda cid: contracts[cid].title,
    )
    contract = contracts[contract_id]
    with st.spinner("Retrieving evidence..."):
        results, seconds = retrieve(contract_id, METHOD, PROCESSED_DIR)
    thresholds = presence_thresholds(METHOD, TEST_RUN, REPORTS_DIR)
    st.caption(
        f"{contract.contract_id} · {len(contract.text):,} characters · retrieval with "
        f"`{METHOD}` took {seconds:.2f} s (cached after the first view)"
    )
    for section in ReportSection:
        st.subheader(section.value)
        for clause in (c for c in ClauseType if SECTION_OF[c] == section):
            show_clause(clause, results[clause], thresholds[clause])
    with st.expander("Full contract text"):
        st.code(contract.text, language=None, wrap_lines=True)


def benchmark_tab() -> None:
    st.markdown(
        "Retrieval quality on CUAD, searching each contract for each clause type. "
        "Brackets are 95% confidence intervals from a paired, contract-level bootstrap; "
        "ΔR@5 is the paired difference against BM25. Latency is per query on one CPU machine "
        "with chunk embeddings cached."
    )
    st.subheader("Test split — evaluated once, all methods frozen")
    st.markdown(markdown_table(load_rows(TEST_RUN, REPORTS_DIR)))
    st.subheader("Dev split — used to compare methods before the test run")
    st.markdown(markdown_table(load_rows(DEV_RUNS, REPORTS_DIR)))
    st.markdown(
        "† Query expanded after studying dev failures: tuned on dev, so it is reported apart "
        "from the fair comparison. Its gain over BM25 roughly halved from dev to test.\n\n"
        "**Presence AUROC** measures how well the top-1 score separates contracts that contain "
        "the clause from those that do not. Hybrid retrieval without reranking scores poorly "
        "because reciprocal rank fusion scores encode ranks, not content.\n\n"
        "Full method and per-clause results: `docs/EVALUATION.md`."
    )


def main() -> None:
    st.set_page_config(page_title="Retrieval Labs", layout="wide")
    st.title("Retrieval Labs")
    st.markdown(
        "Finds the passages of a contract that support five clause types — governing law, "
        "termination for convenience, cap on liability, renewal term and renewal notice period "
        "— and measures how well different retrieval methods do it."
    )
    st.info(
        "Technical demonstration, not legal advice. The demo shows where the system looks; "
        "it does not interpret clauses. *Insufficient evidence* means the best passage scored "
        "below a threshold, not that the clause is proven absent."
    )
    with st.sidebar:
        st.markdown(f"**Retriever:** `{METHOD}`")
        if METHOD == APPLICATION_METHOD:
            st.caption(
                "BM25 and bge-small-en-v1.5 fused by reciprocal rank fusion; the top 20 "
                "rescored by the ms-marco-MiniLM-L6-v2 cross-encoder. 150-word chunks, "
                "50-word overlap."
            )
        st.markdown(
            "**LLM extraction** is implemented but not yet evaluated, so it is not part of "
            "this demo."
        )
        st.markdown(
            "**Data:** contracts from the [Contract Understanding Atticus Dataset (CUAD)]"
            "(https://www.atticusprojectai.org/cuad) by The Atticus Project, licensed "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
        )
    evidence, benchmark = st.tabs(["Evidence by clause", "Benchmark"])
    with evidence:
        evidence_tab()
    with benchmark:
        benchmark_tab()


main()
