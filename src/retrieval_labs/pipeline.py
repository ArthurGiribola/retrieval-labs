"""End-to-end contract analysis: chunk -> index -> retrieve -> select evidence -> extract.

Takes a ``Contract`` only; it has no access to datasets or gold annotations.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import retrieval_labs
from retrieval_labs.clauses import ClauseType, ReportSection
from retrieval_labs.documents import Contract
from retrieval_labs.extraction.budget import SpendLedger
from retrieval_labs.extraction.evidence import build_result, select_passages
from retrieval_labs.extraction.llm import Extractor, build_request
from retrieval_labs.extraction.prompt import PROMPT_VERSION
from retrieval_labs.extraction.schema import ClauseResult, LLMUsage, Status
from retrieval_labs.ingestion.chunking import ChunkingConfig, chunk_contract
from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.queries import QUERIES


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunking: ChunkingConfig = ChunkingConfig()
    top_k: int = Field(default=5, gt=0)


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    package_version: str
    provider: str
    model: str
    extractor_settings: dict[str, Any]
    prompt_version: str
    retriever: str
    retriever_params: dict[str, Any]
    chunking: str
    top_k: int


class ContractReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    title: str
    run: RunMetadata
    chunk_count: int
    timings_s: dict[str, float]
    total_usage: LLMUsage
    results: list[ClauseResult]

    def by_section(self) -> dict[ReportSection, list[ClauseResult]]:
        sections: dict[ReportSection, list[ClauseResult]] = defaultdict(list)
        for result in self.results:
            sections[result.section].append(result)
        return dict(sections)


def run_metadata(
    run_id: str, retriever: Retriever, extractor: Extractor, config: PipelineConfig
) -> RunMetadata:
    params = getattr(retriever, "params", None)
    return RunMetadata(
        run_id=run_id,
        package_version=retrieval_labs.__version__,
        provider=extractor.provider,
        model=extractor.model,
        extractor_settings=dict(extractor.settings),
        prompt_version=PROMPT_VERSION,
        retriever=retriever.name,
        retriever_params=params.model_dump() if isinstance(params, BaseModel) else {},
        chunking=config.chunking.label,
        top_k=config.top_k,
    )


def _sum_usage(usages: list[LLMUsage]) -> LLMUsage:
    fields = LLMUsage.model_fields
    return LLMUsage(**{name: sum(getattr(u, name) for u in usages) for name in fields})


def analyze_contract(
    contract: Contract,
    retriever: Retriever,
    extractor: Extractor,
    config: PipelineConfig,
    run_id: str,
    ledger: SpendLedger | None = None,
) -> ContractReport:
    """Analyze one contract. With a ledger, every call is budget-checked before it is made
    and recorded after it returns (including failed or truncated calls)."""
    timings: dict[str, float] = defaultdict(float)
    started = time.perf_counter()

    t = time.perf_counter()
    chunks = chunk_contract(contract, config.chunking)
    timings["chunking"] = time.perf_counter() - t

    t = time.perf_counter()
    index = retriever.build(chunks) if chunks else None
    timings["indexing"] = time.perf_counter() - t

    results: list[ClauseResult] = []
    usages: list[LLMUsage] = []
    for clause_type in ClauseType:
        t = time.perf_counter()
        retrieved = index.search(QUERIES[clause_type], config.top_k) if index else []
        passages = select_passages(retrieved, contract.text)
        timings["retrieval"] += time.perf_counter() - t
        chunk_ids = [r.chunk_id for r in retrieved]

        if not passages:
            results.append(
                ClauseResult(
                    clause_type=clause_type,
                    status=Status.NOT_FOUND,
                    extracted_text=None,
                    normalized_value=None,
                    evidence=[],
                    extraction_notes=["no candidate passages retrieved; no model call made"],
                    retrieved_chunk_ids=[],
                )
            )
            continue

        request = build_request(clause_type, passages)
        if ledger is not None:
            ledger.check(extractor.worst_case_cost_usd(request))
        t = time.perf_counter()
        response = extractor.extract(request)
        timings["extraction"] += time.perf_counter() - t
        if ledger is not None:
            ledger.record(
                contract_id=contract.contract_id,
                clause_type=clause_type.value,
                model=extractor.model,
                usage=response.usage,
            )
        usages.append(response.usage)
        result = build_result(
            clause_type,
            response.answer,
            passages,
            contract.contract_id,
            contract.text,
            chunk_ids,
            failure_note=response.failure,
        )
        results.append(result.model_copy(update={"usage": response.usage}))

    timings["total"] = time.perf_counter() - started
    return ContractReport(
        contract_id=contract.contract_id,
        title=contract.title,
        run=run_metadata(run_id, retriever, extractor, config),
        chunk_count=len(chunks),
        timings_s={name: round(value, 4) for name, value in timings.items()},
        total_usage=_sum_usage(usages),
        results=results,
    )
