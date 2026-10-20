"""When does a retrieved chunk count as relevant to a gold span?

Primary criterion (fixed before any experiment was run, see docs/EVALUATION.md):

    overlap(chunk, span) / min(len(chunk), len(span)) >= 0.5      (lengths in characters)

A short span must lie at least half inside the chunk; a span longer than the chunk must
fill at least half of it. This tolerates gold spans that straddle chunk boundaries.
Two alternatives are reported as sensitivity analysis only.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from retrieval_labs.cuad.gold import GoldSpan


class Criterion(StrEnum):
    MIN_NORMALIZED = "min_normalized_0.5"  # primary
    ANY_OVERLAP = "any_overlap"  # lenient: at least one shared character
    CONTAINMENT = "containment"  # strict: the whole span inside the chunk


PRIMARY = Criterion.MIN_NORMALIZED
MIN_NORMALIZED_THRESHOLD = 0.5


def overlap(start: int, end: int, span: GoldSpan) -> int:
    """Number of characters shared by [start, end) and the span."""
    return max(0, min(end, span.end) - max(start, span.start))


def is_relevant(start: int, end: int, span: GoldSpan, criterion: Criterion = PRIMARY) -> bool:
    shared = overlap(start, end, span)
    if criterion == Criterion.ANY_OVERLAP:
        return shared > 0
    if criterion == Criterion.CONTAINMENT:
        return start <= span.start and span.end <= end
    shortest = min(end - start, span.end - span.start)
    return shortest > 0 and shared / shortest >= MIN_NORMALIZED_THRESHOLD


def relevant_spans(
    start: int, end: int, spans: Sequence[GoldSpan], criterion: Criterion = PRIMARY
) -> list[int]:
    """Indices of the gold spans this chunk is relevant to."""
    return [i for i, span in enumerate(spans) if is_relevant(start, end, span, criterion)]
