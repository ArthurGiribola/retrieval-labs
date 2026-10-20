"""BM25 lexical retrieval, implemented directly (see ENGINEERING_LOG LOG-006).

Scoring for a document d and query q (distinct query terms t):

    score(d, q) = sum_t idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

The idf is Lucene's variant, which is always positive. Each index holds the chunks
of a single contract, where a term often occurs in more than half of the chunks;
the classic Robertson-Sparck Jones idf turns negative there.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from retrieval_labs.ingestion.chunking import Chunk
from retrieval_labs.retrieval.base import RetrievalResult, single_contract_id, to_result

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric runs. Used only for scoring; never alters chunk text."""
    return _TOKEN_RE.findall(text.lower())


class BM25Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = Field(default=1.2, ge=0)  # Lucene defaults
    b: float = Field(default=0.75, ge=0, le=1)


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], params: BM25Params) -> None:
        single_contract_id(chunks)
        self._chunks = list(chunks)
        self._params = params
        self._tfs = [Counter(tokenize(chunk.text)) for chunk in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) or 1.0
        self._df: Counter[str] = Counter()
        for tf in self._tfs:
            self._df.update(tf.keys())

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: set[str], position: int) -> float:
        k1, b = self._params.k1, self._params.b
        tf, length = self._tfs[position], self._lengths[position]
        total = 0.0
        # Fixed order: set iteration order varies with PYTHONHASHSEED and float addition is
        # not associative, so summing in set order made scores process-dependent (LOG-008).
        for term in sorted(query_terms):
            freq = tf.get(term, 0)
            if freq:
                norm = k1 * (1 - b + b * length / self._avgdl)
                total += self.idf(term) * freq * (k1 + 1) / (freq + norm)
        return total

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Top-k chunks with a positive score; ties broken by document order.

        Chunks sharing no term with the query are never returned, so fewer than
        ``top_k`` results means the contract has no further lexical candidates.
        """
        terms = set(tokenize(query))
        scored = [(self.score(terms, i), i) for i in range(len(self._chunks))]
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
        return [
            to_result(self._chunks[i], rank, score)
            for rank, (score, i) in enumerate(ranked[:top_k], start=1)
        ]


class BM25Retriever:
    name = "bm25"

    def __init__(self, params: BM25Params | None = None) -> None:
        self.params = params or BM25Params()

    def build(self, chunks: Sequence[Chunk]) -> BM25Index:
        return BM25Index(chunks, self.params)
