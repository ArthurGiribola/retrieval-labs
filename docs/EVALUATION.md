# Evaluation

How Retrieval Labs is evaluated, and what has been measured so far. Every number here comes
from a stored run under `reports/benchmark/`; nothing is estimated or copied from elsewhere.

## Status

| Component | Status |
|---|---|
| Retrieval metrics (Recall@k, MRR, span coverage) | **Measured** on dev — Phase 3 run `phase3-bm25-dev` |
| Presence/absence at retrieval level | **Measured** on dev (thresholds fit on train) |
| Extraction metrics (EM, token P/R/F1) | Implemented and unit-tested; **pending** the paid pilot |
| Test split | **Not used.** Runs once at the end of Phase 4 with all four methods frozen |

## Evaluation items

One item per (contract, target CUAD label). Dev: 102 contracts × 5 labels = 510 items, of which
239 are answerable (the label has at least one gold span). Ranking metrics use answerable items
only; presence detection uses all 510. Retrieval searches only the item's own contract, through
the same `ContractSearch` code path the application uses, which receives the contract alone;
gold spans are consulted only afterwards to judge the ranking.

## Relevance criterion (fixed before any experiment)

A retrieved chunk is relevant to a gold span when, measured in characters,

    overlap(chunk, span) / min(len(chunk), len(span)) >= 0.5

A short span must lie at least half inside the chunk; a span longer than the chunk must fill at
least half of it. This tolerates gold spans that straddle chunk boundaries, which is common:
on train, 46% of `Cap On Liability`, 40% of `Notice Period To Terminate Renewal` and 38% of
`Renewal Term` spans exceed 51 words, the longest run guaranteed to fit inside one 150/50
chunk. An item is a hit at k if any of the top-k chunks is relevant to any of its gold spans.

Sensitivity analysis (reported, never used to choose configurations): *any overlap* (lenient)
and *containment* (whole span inside the chunk, strict).

## Metrics

- **Recall@k** (k = 1, 3, 5, 10): share of answerable items with a relevant chunk in the top k.
- **MRR@10**: mean of 1 / rank of the first relevant chunk, 0 if none in the top 10.
- **Recall@750 words**: recall when every configuration gets the same amount of text
  (750 words = 15, 7 or 5 chunks of 50, 100 or 150 words), so chunk sizes compare fairly.
- **Span coverage@5**: share of an item's gold spans hit by the top 5 (matters for multi-span
  labels such as `Cap On Liability`).
- **Achievable**: share of items for which some chunk of the contract is relevant at all.
- **Presence detection**: predict "present" when the top-1 BM25 score ≥ a per-label threshold
  fit on train (F1-maximizing); AUROC needs no threshold (0.5 = chance, 1.0 = perfect).
- **Micro** averages pool all items; **macro** averages give each label equal weight.
- Extraction (pending): SQuAD-normalized exact match and token-level P/R/F1 against the best
  of the item's gold spans.

## Phase 3 results — BM25 on dev

Run `reports/benchmark/phase3-bm25-dev/` (commit `3296c55`, clean tree; BM25 k1 1.2, b 0.75;
fixed queries). Byte-identical outputs across two runs with different `PYTHONHASHSEED`.

| Chunking (words/overlap) | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@750 words | Span cov@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50/17 | 0.736 | 0.866 | 0.900 | 0.950 | 0.810 | 0.958 | 0.830 |
| 100/33 | 0.795 | 0.891 | 0.916 | 0.962 | 0.849 | 0.950 | 0.876 |
| 150/50 | 0.787 | 0.904 | 0.937 | 0.958 | 0.851 | 0.937 | 0.907 |

239 answerable items: one item is 0.42 percentage points, so differences of one or two items
(e.g. 100/33 vs 150/50 on R@1 or MRR) are not meaningful. Achievable = 1.000 for all three.

Per label, 150/50 (full tables for all configurations in the run's `report.md`):

| Label | Items | R@1 | R@5 | MRR@10 | R@5 containment | Presence AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Governing Law | 89 | 0.955 | 1.000 | 0.978 | 0.944 | 0.971 |
| Termination For Convenience | 33 | 0.758 | 0.970 | 0.834 | 0.879 | 0.741 |
| Cap On Liability | 64 | 0.562 | 0.781 | 0.664 | 0.656 | 0.887 |
| Renewal Term | 32 | 0.781 | 1.000 | 0.870 | 0.938 | 0.860 |
| Notice Period To Terminate Renewal | 21 | 0.810 | 1.000 | 0.877 | 0.952 | 0.748 |

Presence detection, pooled over labels (150/50): precision 0.706, recall 0.874, F1 0.781;
macro AUROC 0.842.

Latency on the development machine (Windows 11, Intel64 Family 6 Model 165, Python 3.12.15),
150/50: index build median 5.3 ms per contract (p95 29.1 ms); query median 0.29 ms (p95 1.2 ms).

### Observations

1. **Ceiling effect (Addendum 21.6) confirmed.** Recall@10 is 0.95–0.96 for every
   configuration; Recall@1 and MRR separate them more.
2. **Smaller chunks did not help BM25 here.** 50/17 is worst at fixed k (R@1 0.736). At an equal
   word budget the order reverses slightly (0.958 vs 0.937), within a few items.
3. **Containment shows why the criterion matters.** With 50-word chunks, R@5 under containment
   drops to 0.234–0.551 per label: most gold spans cannot fit in one chunk.
4. **`Cap On Liability` is the hard label** (R@5 0.781, MRR 0.664), and its misses have a
   specific cause (below).
5. **Presence from the BM25 score works for Governing Law** (AUROC 0.97) but is weak for
   Termination for Convenience and the renewal notice period (AUROC ≈ 0.74–0.82; precision
   0.40–0.48), whose vocabulary ("terminate", "notice", "days") appears in most contracts anyway.

## Failure analysis (150/50, misses at k = 5)

All 15 misses were read manually (dev only). 14 are `Cap On Liability`, 1 is
`Termination For Convenience`.

| Category | Count | Example (abridged) |
|---|---:|---|
| Exclusion of damage types annotated as Cap On Liability | 8 | "NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES..." |
| "Sole remedy" clauses annotated as Cap On Liability | 2 | "The sole and exclusive remedy for any breach... shall be as set forth in Article 8..." |
| Liability allocation clauses (share of loss, risk of loss) | 2 | "Vendor shall be liable to pay only its Proportionate Share of the Covered Loss..." |
| Time limit to bring claims, worded differently from the query | 1 | "neither Party may bring a claim... more than two years after the cause of action accrues" |
| Liquidated damages statement | 1 | "LIQUIDATED DAMAGES PAYABLE BY YOU... ARE NOT PUNITIVE OR EXEMPLARY DAMAGES." |
| Vocabulary mismatch, Termination for Convenience | 1 | "Either Party may terminate this agreement for non-cause with a sixty (60) written notice." |

Main finding: in practice, CUAD's `Cap On Liability` annotations are broader than its own
description ("a cap on liability... maximum amount for recovery or time limitation to bring
claims"). Annotators also mark exclusions of indirect/consequential damages, sole-remedy
clauses and liability allocations. The query, written from the description, targets amounts
and caps, so lexical retrieval misses those clauses. See LOG-009.

Queries were kept frozen in Phase 3. Any revision belongs to Phase 4, must be justified on dev,
documented, and applied identically to all four retrieval methods.

## Limitations

- Dev split only (102 contracts); dev contracts are longer than test contracts (LOG-004).
- Small per-label samples (21–89 answerable items); no confidence intervals yet.
- Relevance is judged against CUAD spans, which inherit annotator decisions (LOG-003, LOG-009).
- Presence detection uses the raw top-1 BM25 score, which is not calibrated across contracts.
- Latency is from one development machine, single process.
