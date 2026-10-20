"""Retrieval and presence-detection metrics. Pure functions over ranks and scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def first_hit_rank(hits: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None if none is relevant."""
    return next((rank for rank, hit in enumerate(hits, start=1) if hit), None)


def hit_at_k(first_rank: int | None, k: int) -> bool:
    return first_rank is not None and first_rank <= k


def reciprocal_rank(first_rank: int | None, cutoff: int = 10) -> float:
    """1 / rank of the first relevant result, or 0 if there is none within the cutoff."""
    return 1.0 / first_rank if hit_at_k(first_rank, cutoff) else 0.0


def span_coverage(spans_per_rank: Sequence[Sequence[int]], n_spans: int, k: int) -> float:
    """Fraction of the gold spans hit by at least one of the top-k results."""
    if n_spans == 0:
        raise ValueError("span coverage is undefined without gold spans")
    covered = {i for spans in spans_per_rank[:k] for i in spans}
    return len(covered) / n_spans


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half).

    Equals the area under the ROC curve; 0.5 is chance. None if a class is empty.
    """
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positive_scores) * len(negative_scores))


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def confusion(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Confusion:
    """Predict present when score >= threshold."""
    tp = fp = fn = tn = 0
    for score, present in zip(scores, labels, strict=True):
        predicted = score >= threshold
        if predicted and present:
            tp += 1
        elif predicted:
            fp += 1
        elif present:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def fit_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Threshold among the observed scores that maximizes F1; the lowest one on ties."""
    if not any(labels):
        raise ValueError("cannot fit a presence threshold without positive examples")
    candidates = sorted(set(scores))
    return max(candidates, key=lambda t: (confusion(scores, labels, t).f1, -t))
