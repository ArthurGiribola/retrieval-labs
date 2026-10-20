# Status

_Last updated: 2026-10-20_

## Current phase

**Phase 3 — Initial Evaluation: retrieval part complete.** BM25 retrieval and retrieval-level
presence detection are measured on dev (`docs/EVALUATION.md`). LLM extraction metrics are
implemented and unit-tested but **pending the deferred paid pilot**. Waiting for the owner's
choice of chunk configuration and authorization for Phase 4.

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
| `uv run pytest` | 177 passed (includes 6 `data` tests and the hash-seed subprocess test) |
| `uv run ruff check .` / `ruff format --check .` | Clean |
| `run_benchmark.py` twice, `PYTHONHASHSEED` 11 and 97 | All outputs byte-identical (except run name) |
| `run_pipeline.py --dev-first 3 --text-dir data/fixtures/adversarial --dry-run` | Unchanged after Phase 3 refactor: worst case US$ 0.0937 |

Project LLM spend so far: US$ 0.00.

## Important technical decisions

1. Relevance criterion fixed before experiments; any-overlap and containment are sensitivity only.
2. Phase 3 uses dev only; test runs once at the end of Phase 4 with all four methods frozen.
3. Presence thresholds fit on train, reported on dev.
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

1. Phase 4 plan (dense, hybrid, hybrid + reranking) — awaiting owner review, including model
   names, sizes and licenses before any download.
2. **Deferred: paid pilot** — `ANTHROPIC_API_KEY` in `.env`, then
   `uv run --env-file .env python scripts/run_pipeline.py --dev-first 3 --text-dir
   data/fixtures/adversarial --provider anthropic --max-usd 0.25`. After it: run the extraction
   metrics, compare adversarial results with `EXPECTED.md`, measured tokens with the bound.
   Note for the pilot: the extraction definition of Cap On Liability may score as errors the
   damage exclusions CUAD counts as caps (LOG-009).

## Known problems / risks

- Dev contracts are longer than test contracts (LOG-004); dev and test results not comparable.
- Small per-label samples (21–89 answerable dev items); no confidence intervals yet.
- Retrieval-level presence detection is weak for Termination for Convenience and the renewal
  notice period (AUROC ≈ 0.74–0.82).
- Long gold spans straddle chunks; extraction can only quote what retrieved passages contain.
- Verification proves a quote exists, not that it is a legitimate clause (adversarial pilot).
- Near-duplicate contracts are not detected; LLM outputs are not deterministic.
- Playbook perspective for renewal notice undefined (Phase 5).
