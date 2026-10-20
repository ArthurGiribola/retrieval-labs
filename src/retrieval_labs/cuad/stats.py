"""Descriptive statistics of the prepared dataset, per split and per target label."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from retrieval_labs.cuad.gold import GoldAnnotation, GoldSpan
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.splits import Split
from retrieval_labs.documents import Contract


def describe(values: Sequence[int]) -> dict[str, float]:
    """min / median / mean / p95 (nearest rank) / max; empty dict for no values."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def overlapping_pairs(spans: Sequence[GoldSpan]) -> int:
    """Number of span pairs that share at least one character (spans sorted by start)."""
    return sum(1 for i, a in enumerate(spans) for b in spans[i + 1 :] if b.start < a.end)


def compute_stats(
    contracts: Sequence[Contract],
    gold: Sequence[GoldAnnotation],
    assignment: Mapping[str, Split],
) -> dict[str, Any]:
    per_split_chars: dict[Split, list[int]] = {split: [] for split in Split}
    for contract in contracts:
        per_split_chars[assignment[contract.contract_id]].append(len(contract.text))

    labels: dict[str, dict[str, Any]] = {}
    for label in CuadLabel:
        labels[label.value] = {}
        for split in Split:
            selected = [g for g in gold if g.label == label and assignment[g.contract_id] == split]
            answerable = [g for g in selected if g.is_answerable]
            labels[label.value][split.value] = {
                "contracts": len(selected),
                "contracts_with_answer": len(answerable),
                "contracts_without_answer": len(selected) - len(answerable),
                "contracts_with_multiple_spans": sum(1 for g in answerable if len(g.spans) > 1),
                "total_spans": sum(len(g.spans) for g in answerable),
                "max_spans_in_one_contract": max((len(g.spans) for g in answerable), default=0),
                "overlapping_span_pairs": sum(overlapping_pairs(g.spans) for g in answerable),
                "span_chars": describe([len(s.text) for g in answerable for s in g.spans]),
            }

    counts = Counter(assignment.values())
    return {
        "contracts_per_split": {split.value: counts[split] for split in Split},
        "contract_chars_per_split": {
            split.value: describe(values) for split, values in per_split_chars.items()
        },
        "labels": labels,
    }
