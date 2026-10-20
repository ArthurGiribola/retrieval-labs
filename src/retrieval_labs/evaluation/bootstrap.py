"""Paired, contract-level bootstrap confidence intervals.

Items from the same contract are not independent (one contract yields five items), so
resampling draws whole contracts with replacement. Comparisons between methods reuse the
same resampled contracts for both methods (paired), which gives an interval for the
difference itself rather than two overlapping intervals.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

N_RESAMPLES = 1000
SEED = 42
CONFIDENCE = 0.95

ByContract = Mapping[str, Sequence[float]]  # contract_id -> per-item values (e.g. hit@1 as 0/1)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _mean(groups: Sequence[Sequence[float]]) -> float:
    values = [v for group in groups for v in group]
    return sum(values) / len(values)


def _percentiles(stats: list[float], confidence: float) -> tuple[float, float]:
    stats.sort()
    alpha = 1 - confidence
    low = stats[max(0, math.floor(alpha / 2 * len(stats)))]
    high = stats[min(len(stats) - 1, math.ceil((1 - alpha / 2) * len(stats)) - 1)]
    return low, high


def _resamples(contracts: list[str], n: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [[rng.choice(contracts) for _ in contracts] for _ in range(n)]


def bootstrap_ci(
    values: ByContract, n: int = N_RESAMPLES, seed: int = SEED, confidence: float = CONFIDENCE
) -> Interval:
    contracts = sorted(c for c, v in values.items() if v)
    if not contracts:
        raise ValueError("no values to bootstrap")
    stats = [_mean([values[c] for c in sample]) for sample in _resamples(contracts, n, seed)]
    low, high = _percentiles(stats, confidence)
    return Interval(_mean([values[c] for c in contracts]), low, high)


def paired_difference_ci(
    a: ByContract,
    b: ByContract,
    n: int = N_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> Interval:
    """Interval for mean(a) - mean(b) over the same items of the same contracts."""
    if {c: len(v) for c, v in a.items()} != {c: len(v) for c, v in b.items()}:
        raise ValueError("paired comparison needs the same items for both methods")
    contracts = sorted(c for c, v in a.items() if v)
    if not contracts:
        raise ValueError("no values to bootstrap")
    stats = [
        _mean([a[c] for c in sample]) - _mean([b[c] for c in sample])
        for sample in _resamples(contracts, n, seed)
    ]
    low, high = _percentiles(stats, confidence)
    estimate = _mean([a[c] for c in contracts]) - _mean([b[c] for c in contracts])
    return Interval(estimate, low, high)
