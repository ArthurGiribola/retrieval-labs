# Evaluation

How Retrieval Labs is evaluated, and what has been measured so far. Every number here comes
from a stored run under `reports/benchmark/`; nothing is estimated or copied from elsewhere.

## Status

| Component | Status |
|---|---|
| Retrieval metrics (Recall@k, MRR, span coverage) | **Measured** on dev (Phase 3 `phase3-bm25-dev`, Phase 4 `phase4-dev`) and once on test (`phase4-test`) |
| Presence/absence at retrieval level | **Measured** on dev and test (thresholds fit on train) |
| Extraction metrics (EM, token P/R/F1) | Implemented and unit-tested; **pending** the paid pilot |
| Test split | **Used once** at the end of Phase 4 (`phase4-test`), five frozen methods |

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

## Phase 4 results — retrieval comparison (dev, then test once)

Setup frozen before the test run: chunking 150/50; queries unchanged from Phase 3 (LOG-010);
dense `BAAI/bge-small-en-v1.5@5c38ec7c`; hybrid = reciprocal rank fusion of BM25 and dense
(k = 60); reranking = `cross-encoder/ms-marco-MiniLM-L6-v2@233902d2` over the hybrid top 20.
Intervals: 1000 contract-level bootstrap resamples (seed 42), 95% percentile; differences are
paired against BM25 on the same contracts. Presence thresholds fit on train.

`bm25-expanded †` uses a query expanded from the Phase 3 **dev** failure analysis. It is
reported separately: on dev it is optimistic by construction; only its test row is a fair
estimate.

Runs: `reports/benchmark/phase4-dev/`, `reports/benchmark/phase4-dev-bm25-expanded/`,
`reports/benchmark/phase4-test/` (test run at commit `022de41`, clean tree). Each was repeated
with a different `PYTHONHASHSEED`; `summary.json`, item files and failure samples were
byte-identical.

### Test split (102 contracts, 198 answerable items, micro average)

| Method | R@1 | R@5 | MRR@10 | ΔR@1 vs BM25 | ΔR@5 vs BM25 | ΔMRR@10 vs BM25 | Presence AUROC (macro) |
|---|---|---|---|---|---|---|---:|
| bm25 | 0.833 [0.766, 0.891] | 0.944 [0.906, 0.975] | 0.881 [0.828, 0.923] | — | — | — | 0.866 |
| dense | 0.702 [0.637, 0.768] | 0.899 [0.848, 0.940] | 0.784 [0.734, 0.834] | -0.131 [-0.200, -0.058] | -0.045 [-0.089, +0.000] | -0.097 [-0.146, -0.044] | 0.818 |
| hybrid | 0.813 [0.750, 0.871] | 0.955 [0.923, 0.979] | 0.869 [0.819, 0.912] | -0.020 [-0.063, +0.029] | +0.010 [-0.006, +0.031] | -0.012 [-0.042, +0.022] | 0.612 |
| hybrid+rerank | 0.869 [0.815, 0.914] | 0.980 [0.959, 0.995] | 0.918 [0.883, 0.947] | +0.035 [-0.015, +0.084] | +0.035 [+0.010, +0.067] | +0.037 [+0.005, +0.070] | 0.919 |
| bm25-expanded † | 0.879 [0.824, 0.929] | 0.975 [0.944, 0.995] | 0.919 [0.877, 0.953] | +0.045 [+0.020, +0.077] | +0.030 [+0.009, +0.058] | +0.038 [+0.020, +0.061] | 0.885 |

### Dev split (102 contracts, 239 answerable items, micro average)

| Method | R@1 | R@5 | MRR@10 | ΔR@1 vs BM25 | ΔR@5 vs BM25 | ΔMRR@10 vs BM25 | Presence AUROC (macro) |
|---|---|---|---|---|---|---|---:|
| bm25 | 0.787 [0.732, 0.838] | 0.937 [0.907, 0.966] | 0.851 [0.810, 0.888] | — | — | — | 0.842 |
| dense | 0.611 [0.543, 0.677] | 0.866 [0.813, 0.915] | 0.720 [0.670, 0.773] | -0.176 [-0.235, -0.116] | -0.071 [-0.121, -0.027] | -0.130 [-0.173, -0.083] | 0.796 |
| hybrid | 0.745 [0.688, 0.800] | 0.937 [0.904, 0.965] | 0.828 [0.786, 0.870] | -0.042 [-0.087, +0.000] | +0.000 [-0.031, +0.033] | -0.022 [-0.056, +0.008] | 0.597 |
| hybrid+rerank | 0.791 [0.730, 0.847] | 0.962 [0.934, 0.987] | 0.868 [0.827, 0.906] | +0.004 [-0.038, +0.045] | +0.025 [+0.000, +0.052] | +0.018 [-0.010, +0.044] | 0.883 |
| bm25-expanded † | 0.870 [0.825, 0.916] | 0.992 [0.978, 1.000] | 0.918 [0.888, 0.948] | +0.084 [+0.051, +0.119] | +0.054 [+0.027, +0.083] | +0.067 [+0.044, +0.094] | 0.852 |

### Cap On Liability (the label with the vocabulary gap, LOG-009/010)

| Method | Dev R@1 | Dev R@5 | Test R@1 | Test R@5 |
|---|---:|---:|---:|---:|
| bm25 | 0.562 | 0.781 | 0.614 | 0.841 |
| dense | 0.578 | 0.844 | 0.682 | 0.886 |
| hybrid | 0.656 | 0.906 | 0.705 | 0.864 |
| hybrid+rerank | 0.656 | 0.906 | 0.773 | 0.932 |
| bm25-expanded † | 0.906 | 0.969 | 0.841 | 0.977 |

### Latency (one CPU development machine, single process)

Primary numbers are from the warm-cache repetition (all chunk embeddings cached), stored as
`timings-warm.json`. Query time includes query encoding and, for reranking, scoring 20
candidates. Model load: embedder 16.0 s, reranker 0.7 s (test, warm run).

| Method | Test query median ms | Test query p95 ms | Test index build median / p95 ms |
|---|---:|---:|---:|
| bm25 | 0.223 | 1.348 | 3.1 / 21.0 |
| dense | 16.917 | 21.255 | 3.5 / 16.9 |
| hybrid | 17.929 | 23.388 | 5.3 / 33.0 |
| hybrid+rerank | 431.298 | 713.122 | 6.1 / 36.2 |
| bm25-expanded † | 0.310 | 1.788 | 4.3 / 29.9 |

First execution (cold cache, `timings.json`): on test no contract was cached, so dense index
build (embedding every chunk) took median 1489 ms and p95 11086 ms per contract. Query medians
were not slower cold (bm25 0.221, dense 15.674, hybrid 15.947, hybrid+rerank 407.745,
bm25-expanded 0.256 ms); differences between the two runs are timing noise. On dev, 34 of 102 contracts were uncached in the
first run (dense index build p95 7415 ms).

### Observations

- **No fair method beats BM25 by a wide margin.** Dense alone is significantly worse on both
  splits. Hybrid ties BM25 on ranking.
- **Hybrid + reranking is the best fair method.** On dev its gain was not significant; on test
  R@5 (+0.035) and MRR@10 (+0.037) are significant, R@1 is not. It costs ~430 ms per query on
  CPU versus 0.2 ms for BM25.
- **The dev-tuned query held up on test, but smaller.** `bm25-expanded` gains roughly halved
  from dev to test (ΔR@1 +0.084 → +0.045; ΔR@5 +0.054 → +0.030), as expected from tuning on dev.
  On test its point estimates are close to hybrid + reranking (no paired comparison between
  the two was run) and it is ~1400× faster per query (LOG-013).
- **Hybrid scores cannot drive presence detection** (AUROC 0.597 dev, 0.612 test): RRF scores
  encode ranks only (LOG-012). Reranking restores a content score and gives the best presence
  AUROC on both splits.
- **Test numbers are higher than dev for every method**, consistent with test contracts being
  shorter (LOG-004). Compare methods within a split, not across splits.

## Limitations

- Phase 3 used dev only; Phase 4 used test once. Dev contracts are longer than test contracts
  (LOG-004), so dev and test absolute numbers are not comparable.
- Small per-label samples (16–89 answerable items per split); intervals are wide per label and
  are reported only for micro averages.
- Relevance is judged against CUAD spans, which inherit annotator decisions (LOG-003, LOG-009).
- Presence detection uses raw top-1 scores, which are not calibrated across contracts; RRF
  scores are unusable for it (LOG-012).
- Latency is from one development machine, single process, CPU only.
