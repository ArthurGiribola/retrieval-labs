"""Evaluate retrieval on CUAD: one item per (contract, target label).

Retrieval runs through ``ContractSearch`` (the application's own retrieval path), which
receives only the contract. Gold spans are consulted afterwards, to judge the ranking.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from retrieval_labs.clauses import ClauseType
from retrieval_labs.cuad.gold import GoldAnnotation
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.documents import Contract
from retrieval_labs.evaluation.labels import CUAD_LABEL_OF
from retrieval_labs.evaluation.relevance import PRIMARY, Criterion, relevant_spans
from retrieval_labs.evaluation.retrieval_metrics import (
    Confusion,
    auroc,
    confusion,
    fit_threshold,
    hit_at_k,
    mean,
    reciprocal_rank,
    span_coverage,
)
from retrieval_labs.ingestion.chunking import ChunkingConfig
from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.contract_search import ContractSearch

CUTOFFS = (1, 3, 5, 10)
MRR_CUTOFF = 10
DEPTH = 20  # results kept per item; enough for a 750-word budget with 50-word chunks
WORD_BUDGET = 750  # 5 chunks of 150 words: the pipeline's default context budget


class ItemRecord(BaseModel):
    """Outcome of one (contract, label) retrieval. Contains IDs and numbers, no text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    label: CuadLabel
    answerable: bool
    n_gold_spans: int
    n_chunks: int
    top1_score: float
    ranked_chunk_ids: list[str]
    scores: list[float]
    spans_per_rank: list[list[int]]  # gold span indices each result is relevant to (primary)
    first_rank: dict[Criterion, int | None]
    achievable: dict[Criterion, bool]  # some chunk of the contract is relevant at all


def evaluate_contract(
    contract: Contract,
    gold: dict[CuadLabel, GoldAnnotation],
    retriever: Retriever,
    chunking: ChunkingConfig,
) -> tuple[list[ItemRecord], dict[str, Any]]:
    search = ContractSearch(contract, retriever, chunking)  # built without gold
    query_times: list[float] = []
    records = []
    for clause_type in ClauseType:
        started = time.perf_counter()
        results = search.search(clause_type, DEPTH)
        query_times.append(time.perf_counter() - started)

        label = CUAD_LABEL_OF[clause_type]
        spans = gold[label].spans
        first_rank: dict[Criterion, int | None] = {}
        achievable: dict[Criterion, bool] = {}
        for criterion in Criterion:
            ranks = [
                rank
                for rank, r in enumerate(results, start=1)
                if relevant_spans(r.start, r.end, spans, criterion)
            ]
            first_rank[criterion] = ranks[0] if ranks else None
            achievable[criterion] = any(
                relevant_spans(c.start, c.end, spans, criterion) for c in search.chunks
            )
        records.append(
            ItemRecord(
                contract_id=contract.contract_id,
                label=label,
                answerable=bool(spans),
                n_gold_spans=len(spans),
                n_chunks=len(search.chunks),
                top1_score=round(results[0].score, 6) if results else 0.0,
                ranked_chunk_ids=[r.chunk_id for r in results],
                scores=[round(r.score, 6) for r in results],
                spans_per_rank=[relevant_spans(r.start, r.end, spans) for r in results],
                first_rank=first_rank,
                achievable=achievable,
            )
        )
    timings = {**search.timings_s, "queries": query_times}
    return records, timings


def evaluate_split(
    contracts: Sequence[Contract],
    gold: Sequence[GoldAnnotation],
    retriever: Retriever,
    chunking: ChunkingConfig,
) -> tuple[list[ItemRecord], dict[str, list[float]]]:
    by_contract: dict[str, dict[CuadLabel, GoldAnnotation]] = defaultdict(dict)
    for annotation in gold:
        by_contract[annotation.contract_id][annotation.label] = annotation
    records: list[ItemRecord] = []
    timings: dict[str, list[float]] = defaultdict(list)
    for contract in sorted(contracts, key=lambda c: c.contract_id):
        items, contract_timings = evaluate_contract(
            contract, by_contract[contract.contract_id], retriever, chunking
        )
        records.extend(items)
        timings["index_build_s"].append(contract_timings["chunking"] + contract_timings["indexing"])
        timings["query_s"].extend(contract_timings["queries"])
    return records, dict(timings)


# --- summaries ----------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def ranking_metrics(records: Sequence[ItemRecord], chunking: ChunkingConfig) -> dict[str, Any]:
    """Metrics over answerable items only."""
    items = [r for r in records if r.answerable]
    budget_k = WORD_BUDGET // chunking.window_words
    primary = [r.first_rank[PRIMARY] for r in items]
    metrics: dict[str, Any] = {
        "answerable_items": len(items),
        **{f"recall@{k}": _round(mean([hit_at_k(f, k) for f in primary])) for k in CUTOFFS},
        f"mrr@{MRR_CUTOFF}": _round(mean([reciprocal_rank(f, MRR_CUTOFF) for f in primary])),
        f"recall@{WORD_BUDGET}words": _round(mean([hit_at_k(f, budget_k) for f in primary])),
        "results_in_word_budget": budget_k,
        "span_coverage@5": _round(
            mean([span_coverage(r.spans_per_rank, r.n_gold_spans, 5) for r in items])
        ),
        "achievable": _round(mean([r.achievable[PRIMARY] for r in items])),
        "sensitivity": {
            criterion.value: {
                "recall@5": _round(mean([hit_at_k(r.first_rank[criterion], 5) for r in items])),
                f"mrr@{MRR_CUTOFF}": _round(
                    mean([reciprocal_rank(r.first_rank[criterion], MRR_CUTOFF) for r in items])
                ),
            }
            for criterion in Criterion
            if criterion != PRIMARY
        },
    }
    return metrics


def summarize_ranking(records: Sequence[ItemRecord], chunking: ChunkingConfig) -> dict[str, Any]:
    per_label = {
        label.value: ranking_metrics([r for r in records if r.label == label], chunking)
        for label in CuadLabel
    }
    scalar = [k for k, v in next(iter(per_label.values())).items() if isinstance(v, float)]
    macro = {k: _round(mean([m[k] for m in per_label.values()])) for k in scalar}
    return {"micro": ranking_metrics(records, chunking), "macro": macro, "per_label": per_label}


def summarize_presence(
    fit_records: Sequence[ItemRecord], eval_records: Sequence[ItemRecord]
) -> dict[str, Any]:
    """Top-1 score as a presence signal: threshold per label fit on one split, reported on
    another; AUROC needs no threshold."""
    per_label: dict[str, Any] = {}
    results: list[Confusion] = []
    for label in CuadLabel:
        fit = [r for r in fit_records if r.label == label]
        ev = [r for r in eval_records if r.label == label]
        threshold = fit_threshold([r.top1_score for r in fit], [r.answerable for r in fit])
        result = confusion([r.top1_score for r in ev], [r.answerable for r in ev], threshold)
        results.append(result)
        per_label[label.value] = {
            "threshold_fit_on_train": _round(threshold),
            "auroc": _round(
                auroc(
                    [r.top1_score for r in ev if r.answerable],
                    [r.top1_score for r in ev if not r.answerable],
                )
            ),
            **_classification(result),
            "present": sum(r.answerable for r in ev),
            "absent": sum(not r.answerable for r in ev),
        }
    pooled = Confusion(*(sum(getattr(c, f) for c in results) for f in ("tp", "fp", "fn", "tn")))
    aurocs = [m["auroc"] for m in per_label.values() if m["auroc"] is not None]
    return {
        "per_label": per_label,
        "pooled": _classification(pooled),
        "macro_auroc": _round(mean(aurocs)),
    }


def _classification(result: Confusion) -> dict[str, Any]:
    return {
        "precision": _round(result.precision),
        "recall": _round(result.recall),
        "f1": _round(result.f1),
        "confusion": {"tp": result.tp, "fp": result.fp, "fn": result.fn, "tn": result.tn},
    }


def per_contract_values(records: Sequence[ItemRecord]) -> dict[str, dict[str, list[float]]]:
    """Per-item metric values of answerable items, grouped by contract (for the bootstrap)."""
    metrics: dict[str, dict[str, list[float]]] = {
        "recall@1": defaultdict(list),
        "recall@5": defaultdict(list),
        f"mrr@{MRR_CUTOFF}": defaultdict(list),
    }
    for r in records:
        if not r.answerable:
            continue
        first = r.first_rank[PRIMARY]
        metrics["recall@1"][r.contract_id].append(float(hit_at_k(first, 1)))
        metrics["recall@5"][r.contract_id].append(float(hit_at_k(first, 5)))
        metrics[f"mrr@{MRR_CUTOFF}"][r.contract_id].append(reciprocal_rank(first, MRR_CUTOFF))
    return {name: dict(values) for name, values in metrics.items()}


def latency_stats(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "count": len(ordered),
        "mean_ms": round(1000 * statistics.fmean(ordered), 3),
        "median_ms": round(1000 * statistics.median(ordered), 3),
        "p95_ms": round(1000 * p95, 3),
    }
