"""Evidence selection (retrieved chunks -> passages) and verification (quotes -> offsets).

Verification is what makes ``extracted_text`` verbatim by construction: a quote from
the model is only accepted if it can be located in a passage, and the accepted text
is always the contract's own slice at the located offsets, never the model's string.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval_labs.clauses import ClauseType
from retrieval_labs.extraction.schema import ClauseAnswer, ClauseResult, Evidence, Status
from retrieval_labs.retrieval.base import RetrievalResult


@dataclass(frozen=True)
class Passage:
    """Contiguous contract text formed by merging overlapping retrieved chunks."""

    passage_id: str
    start: int
    end: int
    text: str
    chunks: tuple[tuple[str, int, int], ...]  # (chunk_id, start, end)


def select_passages(results: Sequence[RetrievalResult], contract_text: str) -> list[Passage]:
    """Merge overlapping or touching chunks; return passages in document order (P1, P2, ...)."""
    groups: list[list[RetrievalResult]] = []
    for result in sorted(results, key=lambda r: (r.start, r.end)):
        if groups and result.start <= max(r.end for r in groups[-1]):
            groups[-1].append(result)
        else:
            groups.append([result])
    passages = []
    for number, group in enumerate(groups, start=1):
        start, end = group[0].start, max(r.end for r in group)
        passages.append(
            Passage(
                passage_id=f"P{number}",
                start=start,
                end=end,
                text=contract_text[start:end],
                chunks=tuple((r.chunk_id, r.start, r.end) for r in group),
            )
        )
    return passages


# Characters models commonly normalize when copying: quotes, apostrophes, dashes.
_EQUIVALENT = {
    **dict.fromkeys("'‘’`", "['‘’`]"),
    **dict.fromkeys('"“”', '["“”]'),
    **dict.fromkeys("-‐‑–—", "[-‐‑–—]"),
}


def _tolerant_pattern(quote: str) -> re.Pattern[str] | None:
    """Match the quote's words in order, allowing any whitespace between them and
    equivalent quote/dash characters. Letters and digits must match exactly."""
    words = quote.split()
    if not words:
        return None
    parts = ["".join(_EQUIVALENT.get(ch, re.escape(ch)) for ch in word) for word in words]
    return re.compile(r"\s+".join(parts))


def locate_quote(quote: str, passage: Passage) -> tuple[int, int] | None:
    """Absolute [start, end) of the quote's first occurrence in the passage, or None."""
    if not quote.strip():
        return None
    index = passage.text.find(quote)
    if index >= 0:
        return passage.start + index, passage.start + index + len(quote)
    pattern = _tolerant_pattern(quote)
    match = pattern.search(passage.text) if pattern else None
    if match is None:
        return None
    return passage.start + match.start(), passage.start + match.end()


def _chunk_for(passage: Passage, start: int, end: int) -> str:
    """A chunk containing the whole span if one exists, else the chunk where it starts."""
    for chunk_id, c_start, c_end in passage.chunks:
        if c_start <= start and end <= c_end:
            return chunk_id
    return next(cid for cid, c_start, c_end in passage.chunks if c_start <= start < c_end)


def verify_quotes(
    answer: ClauseAnswer, passages: Sequence[Passage], contract_id: str, contract_text: str
) -> tuple[list[Evidence], int]:
    """Verified evidence (deduplicated, in the model's order of relevance) and the number of
    rejected quotes."""
    by_id = {p.passage_id: p for p in passages}
    found: dict[tuple[int, int], Evidence] = {}
    rejected = 0
    for quote in answer.quotes:
        cited = by_id.get(quote.passage_id)
        candidates = [cited, *(p for p in passages if p is not cited)] if cited else passages
        span, passage = None, None
        for passage in candidates:
            span = locate_quote(quote.text, passage)
            if span:
                break
        if span is None or passage is None:
            rejected += 1
            continue
        start, end = span
        if span in found:
            continue
        found[span] = Evidence(
            document_id=contract_id,
            chunk_id=_chunk_for(passage, start, end),
            source_start=start,
            source_end=end,
            exact_quote=contract_text[start:end],
        )
    return list(found.values()), rejected


def build_result(
    clause_type: ClauseType,
    answer: ClauseAnswer | None,
    passages: Sequence[Passage],
    contract_id: str,
    contract_text: str,
    retrieved_chunk_ids: list[str],
    failure_note: str | None = None,
) -> ClauseResult:
    """Turn a model answer into a verified result, downgrading unsupported claims."""
    if answer is None:
        return ClauseResult(
            clause_type=clause_type,
            status=Status.UNCERTAIN,
            extracted_text=None,
            normalized_value=None,
            evidence=[],
            extraction_notes=[failure_note or "no answer from the extractor"],
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    notes = [answer.notes] if answer.notes.strip() else []
    if answer.status == Status.NOT_FOUND:
        if answer.quotes:
            notes.append(f"{len(answer.quotes)} quote(s) ignored because the status is NOT_FOUND")
        return ClauseResult(
            clause_type=clause_type,
            status=Status.NOT_FOUND,
            extracted_text=None,
            normalized_value=None,
            evidence=[],
            extraction_notes=notes,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    evidence, rejected = verify_quotes(answer, passages, contract_id, contract_text)
    status = answer.status
    if rejected:
        notes.append(f"{rejected} quote(s) rejected: not found verbatim in the passages")
    if status == Status.FOUND and not evidence:
        status = Status.UNCERTAIN
        notes.append("downgraded from FOUND: no quote could be verified against the contract")
    return ClauseResult(
        clause_type=clause_type,
        status=status,
        extracted_text=evidence[0].exact_quote if evidence else None,
        normalized_value=answer.value,  # type: ignore[attr-defined]
        evidence=evidence,
        extraction_notes=notes,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )
