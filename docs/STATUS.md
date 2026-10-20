# Status

_Last updated: 2026-10-20_

## Current phase

**Minimum publishable version (brief 21.2, D-002): in progress.** Phase 4 complete and
approved; the application retrieves with hybrid + reranking (D-001). Steps: 1. shared retriever
factory, pipeline defaults to `hybrid+rerank` — done (dev benchmark re-run after the refactor:
item files and run summaries identical); 2. Streamlit demo; 3. tests; 4. English README;
5. close and stop. Phase 5 starts only after owner authorization.

**Phase 4 — Retrieval Comparison: complete.**
Frozen setup: bge-small-en-v1.5 @5c38ec7c, ms-marco-MiniLM-L6-v2 @233902d2, 150/50 chunks,
RRF k=60, 20 rerank candidates, queries unchanged from Phase 3, paired contract-level bootstrap
(95%), "BM25 + expanded query" as a separate dev-tuned row. Full results and observations:
`docs/EVALUATION.md` ("Phase 4 results").

Phase 4 steps:

1. Dependencies (sentence-transformers 6.1.0, torch 2.14.1 CPU, numpy 2.5.3); two-week release
   convention applied from Phase 4 on, earlier lock unchanged (LOG-011).
2. Dense retrieval + embedding cache (`data/cache/embeddings/`, git-ignored).
3. Hybrid RRF. 4. Reranker. 5. Bootstrap + multi-method benchmark.
6. Models: 134.5 MB + 91.8 MB on disk; 0.64% of dev chunks exceed 512 tokens (truncated);
   CPU throughput 24.8 chunks/s (embedding), 40.8 pairs/s (reranker).
7. Dev benchmark: `reports/benchmark/phase4-dev/`. 8. `bm25-expanded` on dev:
   `reports/benchmark/phase4-dev-bm25-expanded/`.
9. Owner authorized the single test run: `reports/benchmark/phase4-test/` (commit `022de41`,
   clean tree, `--final`, five frozen methods). **The test split is now spent.**
10. Every run repeated with `PYTHONHASHSEED=97`: byte-identical `summary.json`, item files and
    failure samples. Warm-cache timings stored as `timings-warm.json` (primary latency, owner
    decision); first-run timings in `timings.json`.

Phase 4 headline (150/50, micro average, 95% bootstrap CI; Δ paired vs BM25):

| Method | Test R@1 | Test R@5 | Test MRR@10 | Test ΔR@5 vs BM25 | Dev ΔR@5 vs BM25 | Test presence AUROC | Test query median (warm) |
|---|---:|---:|---:|---|---|---:|---:|
| bm25 | 0.833 | 0.944 | 0.881 | — | — | 0.866 | 0.22 ms |
| dense | 0.702 | 0.899 | 0.784 | -0.045 [-0.089, +0.000] | -0.071 [-0.121, -0.027] | 0.818 | 16.9 ms |
| hybrid | 0.813 | 0.955 | 0.869 | +0.010 [-0.006, +0.031] | +0.000 [-0.031, +0.033] | 0.612 (LOG-012) | 17.9 ms |
| hybrid+rerank | 0.869 | 0.980 | 0.918 | +0.035 [+0.010, +0.067] | +0.025 [+0.000, +0.052] | 0.919 | 431 ms |
| bm25-expanded † | 0.879 | 0.975 | 0.919 | +0.030 [+0.009, +0.058] | +0.054 [+0.027, +0.083] | 0.885 | 0.31 ms |

† Query tuned on dev; its dev gain roughly halved on test (LOG-013).

Phase 3 (complete): BM25 retrieval and retrieval-level presence measured on dev
(`docs/EVALUATION.md`). LLM extraction metrics implemented and unit-tested, **pending the
deferred paid pilot**.

## Completed components

Phase 0 — Foundation: uv/Python 3.12, pytest, ruff, LF enforcement (LOG-001),
`scripts/inspect_cuad.py`.

Phase 1 — Dataset: `Contract`; `retrieval_labs.cuad` (download + SHA-256, parse, splits
train 306 / dev 102 / test 102, stats, hash-verified store, test-gold lock);
`scripts/prepare_cuad.py`. LOG-002 to LOG-005.

Phase 2 — Functional MVP (complete except the paid pilot): chunking, in-house BM25 (LOG-006),
queries, evidence verification, nonce-framed prompt, `Extractor` interface, `FakeExtractor`,
`AnthropicExtractor` (LOG-007), spend ledger, `pipeline.py`, `scripts/run_pipeline.py`,
adversarial fixtures with pre-registered expectations.

Phase 3 — Initial Evaluation:

- `retrieval/contract_search.py`: the retrieval path shared by the pipeline and the benchmark.
- `evaluation/`: CUAD label mapping; pre-registered relevance criterion
  (overlap / min(chunk, span) ≥ 0.5) with sensitivity criteria; retrieval and presence metrics;
  extraction metrics (EM, token P/R/F1; not yet run); `retrieval_eval.py`.
- `scripts/run_benchmark.py`: deterministic reports in `reports/benchmark/<run>/`; failure
  samples with contract text in git-ignored `data/analysis/`; test split requires `--final`.
- Run `reports/benchmark/phase3-bm25-dev/` (dev, grid 50/17, 100/33, 150/50).
- `docs/EVALUATION.md`; LOG-008 (hash-seed nondeterminism, fixed), LOG-009 (Cap On Liability
  annotation scope).

## Phase 3 headline (dev, BM25, micro over 239 answerable items)

| Chunking | R@1 | R@5 | MRR@10 |
|---|---:|---:|---:|
| 50/17 | 0.736 | 0.900 | 0.810 |
| 100/33 | 0.795 | 0.916 | 0.849 |
| 150/50 | 0.787 | 0.937 | 0.851 |

## Tests executed

| Command | Result |
|---|---|
| `uv run pytest` | 212 passed (includes 6 `data` tests and the hash-seed subprocess test) |
| `uv run ruff check .` / `ruff format --check .` | Clean |
| `run_benchmark.py` twice, `PYTHONHASHSEED` 11 and 97 | All outputs byte-identical (except run name) |
| Phase 4 runs (dev, dev expanded, test) repeated with `PYTHONHASHSEED=97` | `summary.json`, item files, failure samples byte-identical; only timings/provenance differ |
| `run_pipeline.py --dev-first 3 --text-dir data/fixtures/adversarial --dry-run` | Unchanged after Phase 4: worst case US$ 0.0937 |

Project LLM spend so far: US$ 0.00.

## Important technical decisions

1. Relevance criterion fixed before experiments; any-overlap and containment are sensitivity only.
2. Phase 3 used dev only; test was used once at the end of Phase 4 with five frozen methods.
3. Presence thresholds fit on train, reported on dev and test.
4. Benchmark and application share `ContractSearch`, so they cannot drift apart.
5. Queries frozen during Phase 3; failure analysis informs Phase 4.
6. Extraction path must stay runnable (CLAUDE.md); its tests passed unchanged after the refactor.

## Owner decisions (end of Phase 3)

1. **Chunking for Phase 4: 150/50.**
2. **Queries and extraction prompt stay frozen**, aligned with CUAD's written description, so
   Phase 4 measures whether dense/hybrid retrieval close the vocabulary gap BM25 cannot
   (LOG-010). After the main comparison, an optional "BM25 + expanded query" row, marked as
   tuned on dev and reported separately from the fair comparison.

## Pending tasks / next recommended action

1. Finish the minimum publishable version (steps 2–5 above), then stop for owner review.
2. **Deferred: paid pilot** — `ANTHROPIC_API_KEY` in `.env`, then
   `uv run --env-file .env python scripts/run_pipeline.py --dev-first 3 --text-dir
   data/fixtures/adversarial --provider anthropic --max-usd 0.25`. After it: run the extraction
   metrics, compare adversarial results with `EXPECTED.md`, measured tokens with the bound.
   Note for the pilot: the extraction definition of Cap On Liability may score as errors the
   damage exclusions CUAD counts as caps (LOG-009).

## Known problems / risks

- Dev contracts are longer than test contracts (LOG-004); dev and test results not comparable.
- Small per-label samples (16–89 answerable items per split); intervals only on micro averages.
- Retrieval-level presence detection is weak for Termination for Convenience and the renewal
  notice period (BM25 AUROC ≈ 0.74–0.82); RRF scores are unusable for it (LOG-012).
- The test split has been used; any further tuning needs a new held-out set.
- Long gold spans straddle chunks; extraction can only quote what retrieved passages contain.
- Verification proves a quote exists, not that it is a legitimate clause (adversarial pilot).
- Near-duplicate contracts are not detected; LLM outputs are not deterministic.
- Playbook perspective for renewal notice undefined (Phase 5).
