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

## D-003 — Analysis jobs are never retried automatically

- **Date:** 2026-10-21, Phase 6 step 3.
- **Decision:** A job that fails ends in `FAILED` with a short `error_code` and is not run
  again by the system. RQ jobs are enqueued with no retries. A job left `RUNNING` because its
  worker died (crash, kill, container restart) is marked `FAILED` with `worker_lost`, not
  re-queued. Re-running a document means submitting it again, as a new job.
- **Why:** A job may already have made paid LLM calls before it failed or lost its worker;
  any automatic retry could repeat them without anyone deciding to (brief Section 11, budget
  in Section 21.4). The same rule covers lost workers (owner decision).
- **How it works:**
  - A worker claims a job with `UPDATE ... WHERE state = 'PENDING'`, so a duplicate delivery
    of the same job is skipped.
  - Every job has a time limit (`JOB_TIMEOUT_S`, default 900 s); exceeding it ends the job
    with `timeout`.
  - A job still `RUNNING` more than the time limit plus 2 minutes after it started is marked
    `worker_lost` when a worker starts or before it takes its next job.
  - If a lost job's worker is in fact still alive and finishes, it cannot complete the job:
    its chunks and report are rolled back.
  - Error codes: `processing_error`, `timeout`, `budget_exceeded`, `document_corrupted`,
    `document_missing`, `worker_lost`. Tracebacks are logged by the worker and never stored
    or returned.
- **Known limits:**
  - Detection of a lost job waits for a worker to start or take a job.
  - A job whose queue entry is lost before any worker claims it stays `PENDING`; Redis runs
    with append-only persistence to make that unlikely.

## D-004 — Backend before alerts, local only

- **Date:** 2026-10-21, Phase 6 plan.
- **Decision:** Phase 6 (backend and infrastructure) runs before Phase 5 (playbook and
  alerts), which is deferred, not dropped. The backend therefore has no `alerts` table yet;
  Phase 5 adds it with its own migration.
- **Scope (Phase 6 plan, approved by the owner):**
  - Uploads are UTF-8 `.txt` files only (no PDF or DOCX parsing).
  - The stack is for local use: no authentication or rate limiting, and only the API
    publishes a port, on `127.0.0.1`.
  - Model weights are downloaded once into a Docker volume, not baked into the image.
  - Containers run the free extractor only (LOG-018); paid extraction stays on the host,
    where the single spend ledger lives.
- **Not in scope:** deployment and a web upload interface (Phase 7).
