"""Deterministic stand-ins for embedding and reranking models (no downloads, no torch)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _bucket(token: str, dimension: int) -> int:
    # Stable across processes, unlike Python's hash() (see LOG-008).
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % dimension


class HashingEmbedder:
    """Bag-of-words hashed into a fixed dimension, L2-normalized."""

    def __init__(self, dimension: int = 64, name: str = "fake-hashing") -> None:
        self.dimension = dimension
        self.name = name
        self.document_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            vector[_bucket(token, self.dimension)] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls += 1
        return (
            np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dimension))
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class OverlapScorer:
    """Cross-encoder stand-in: score = number of distinct query tokens in the passage."""

    name = "fake-overlap"

    def __init__(self) -> None:
        self.pairs_scored = 0

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        terms = set(_TOKEN.findall(query.lower()))
        self.pairs_scored += len(passages)
        return [float(len(terms & set(_TOKEN.findall(p.lower())))) for p in passages]
