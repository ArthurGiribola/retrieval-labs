# Decisions

Owner decisions that set the project's direction. Each records what was decided, the evidence
behind it and what it rules out. Engineering detail lives in `docs/ENGINEERING_LOG.md`;
measured results in `docs/EVALUATION.md`.

## D-001 — The application retrieves with hybrid + reranking

- **Date:** 2026-10-20, after Phase 4.
- **Decision:** The pipeline (`scripts/run_pipeline.py`) and the demo use `hybrid+rerank`:
  BM25 and `BAAI/bge-small-en-v1.5` fused by reciprocal rank fusion (k = 60), top 20 rescored
  by `cross-encoder/ms-marco-MiniLM-L6-v2`, 150/50 chunks. The default lives in
  `retrieval_labs.retrieval.factory.APPLICATION_METHOD`; `--retriever bm25` remains available.
- **Evidence:** Best method of the fair comparison on the single test run (R@5 0.980, ΔR@5 vs
  BM25 +0.035 [+0.010, +0.067]) and best presence detection (macro AUROC 0.919).
- **Trade-off accepted:** ~430 ms per query on CPU (measured median), so roughly 2 s per
  contract for five clause types (estimated from that median, not measured end to end), plus model loading once and chunk embedding the first time a contract is seen.
  Acceptable for contract review, which is not interactive search.
- **Not chosen:** `bm25-expanded` has close point estimates at ~0.3 ms per query, but its query
  was tuned on dev (LOG-010, LOG-013). Hybrid without reranking cannot drive presence
  detection (LOG-012).

## D-002 — Minimum publishable version before Phase 5

- **Date:** 2026-10-20.
- **Decision:** Following brief Section 21.2, the project gets an English README and a simple
  Streamlit demo before Phase 5. Code is MIT-licensed; CUAD stays CC BY 4.0 and is not
  redistributed (users download it with `scripts/prepare_cuad.py`).
- **Scope:** The demo shows retrieved passages per clause type for a CUAD contract and the
  benchmark tables. LLM extraction stays out of the demo until the paid pilot has run and
  extraction metrics exist; the README lists extraction as implemented but not evaluated.
- **Not in scope:** uploading arbitrary contracts (privacy, brief Section 18; Phase 7),
  deployment, making the repository public (owner's decision).
