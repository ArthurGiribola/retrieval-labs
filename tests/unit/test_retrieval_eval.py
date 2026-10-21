from pathlib import Path

import pytest

from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.parse import make_contract_id, parse_archive
from retrieval_labs.evaluation.relevance import Criterion
from retrieval_labs.evaluation.retrieval_eval import (
    evaluate_split,
    latency_stats,
    summarize_presence,
    summarize_ranking,
)
from retrieval_labs.ingestion.chunking import ChunkingConfig
from retrieval_labs.retrieval.bm25 import BM25Retriever
from synthetic_cuad import ALPHA

pytestmark = pytest.mark.smoke

SMALL = ChunkingConfig(window_words=8, overlap_words=3)


@pytest.fixture
def evaluated(cuad_archive: Path):
    parsed = parse_archive(cuad_archive)
    return evaluate_split(parsed.contracts, parsed.gold, BM25Retriever(), SMALL)


def test_one_record_per_contract_and_label(evaluated) -> None:
    records, timings = evaluated
    assert len(records) == 4 * len(CuadLabel)
    assert len(timings["index_build_s"]) == 4
    assert len(timings["query_s"]) == 4 * len(CuadLabel)


def test_answerable_item_is_found_and_ranked(evaluated) -> None:
    records, _ = evaluated
    law = next(
        r
        for r in records
        if r.contract_id == make_contract_id(ALPHA) and r.label == CuadLabel.GOVERNING_LAW
    )
    assert law.answerable and law.n_gold_spans == 1
    assert law.achievable[Criterion.MIN_NORMALIZED]
    assert law.first_rank[Criterion.MIN_NORMALIZED] is not None
    assert law.scores == sorted(law.scores, reverse=True)
    assert law.top1_score == law.scores[0]


def test_unanswerable_items_have_no_ranks(evaluated) -> None:
    records, _ = evaluated
    for record in records:
        if not record.answerable:
            assert set(record.first_rank.values()) == {None}
            assert record.spans_per_rank == [[] for _ in record.ranked_chunk_ids]


def test_ranking_summary_shape(evaluated) -> None:
    records, _ = evaluated
    summary = summarize_ranking(records, SMALL)
    micro = summary["micro"]
    assert micro["answerable_items"] == sum(r.answerable for r in records)
    assert 0.0 <= micro["recall@1"] <= micro["recall@3"] <= micro["recall@10"] <= 1.0
    assert micro["results_in_word_budget"] == 750 // 8
    assert set(summary["per_label"]) == {label.value for label in CuadLabel}
    assert set(micro["sensitivity"]) == {"any_overlap", "containment"}


def test_presence_summary_reports_every_label(evaluated) -> None:
    records, _ = evaluated
    presence = summarize_presence(records, records)
    for label in CuadLabel:
        entry = presence["per_label"][label.value]
        assert entry["present"] + entry["absent"] == 4
    pooled = presence["pooled"]["confusion"]
    assert sum(pooled.values()) == 4 * len(CuadLabel)


def test_macro_auroc_keeps_zero_values(evaluated) -> None:
    """Regression: an AUROC of 0.0 is a real (worst-case) value, not a missing one."""
    records, _ = evaluated
    inverted = [
        r.model_copy(update={"top1_score": -r.top1_score if r.answerable else 100.0})
        for r in records
    ]
    presence = summarize_presence(inverted, inverted)
    values = [m["auroc"] for m in presence["per_label"].values() if m["auroc"] is not None]
    assert 0.0 in values
    assert presence["macro_auroc"] == pytest.approx(sum(values) / len(values))


def test_evaluation_is_deterministic(cuad_archive: Path) -> None:
    parsed = parse_archive(cuad_archive)
    first, _ = evaluate_split(parsed.contracts, parsed.gold, BM25Retriever(), SMALL)
    again, _ = evaluate_split(list(reversed(parsed.contracts)), parsed.gold, BM25Retriever(), SMALL)
    assert first == again
    assert summarize_ranking(first, SMALL) == summarize_ranking(again, SMALL)


def test_latency_stats() -> None:
    stats = latency_stats([0.001, 0.002, 0.003, 0.004])
    assert stats == {"count": 4, "mean_ms": 2.5, "median_ms": 2.5, "p95_ms": 4.0}
