"""Extraction metrics: exact match and token-level precision/recall/F1 against gold spans.

Implemented ahead of the paid pilot so it ends with numbers; not yet run on model output.

Normalization follows the SQuAD convention (lowercase, strip punctuation and the articles
a/an/the, collapse whitespace). With several acceptable gold spans, a prediction is scored
against the gold span that gives it the highest F1. Only answerable items are scored here;
absence detection is evaluated separately as classification.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_PUNCTUATION = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—"}
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


@dataclass(frozen=True)
class TokenScore:
    precision: float
    recall: float
    f1: float


def token_score(prediction: str, gold: str) -> TokenScore:
    predicted, expected = tokens(prediction), tokens(gold)
    if not predicted or not expected:
        same = predicted == expected
        return TokenScore(float(same), float(same), float(same))
    shared = sum((Counter(predicted) & Counter(expected)).values())
    if shared == 0:
        return TokenScore(0.0, 0.0, 0.0)
    precision, recall = shared / len(predicted), shared / len(expected)
    return TokenScore(precision, recall, 2 * precision * recall / (precision + recall))


def exact_match(prediction: str | None, golds: Sequence[str]) -> float:
    if not golds:
        raise ValueError("exact match is only defined for answerable items")
    if prediction is None:
        return 0.0
    return float(any(normalize(prediction) == normalize(g) for g in golds))


def best_token_score(prediction: str | None, golds: Sequence[str]) -> TokenScore:
    """Score against the best-matching gold span; a missing prediction scores zero."""
    if not golds:
        raise ValueError("token scores are only defined for answerable items")
    if prediction is None:
        return TokenScore(0.0, 0.0, 0.0)
    return max((token_score(prediction, g) for g in golds), key=lambda s: s.f1)
