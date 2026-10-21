# Status

_Last updated: 2026-10-21_

## Current phase

**Phase 6 — Backend and Infrastructure: in progress** (plan approved 2026-10-21; Phase 5
deferred by the owner, so no `alerts` table yet; uploads are UTF-8 `.txt` only; model
weights will be downloaded once into a Docker volume, after owner approval). The owner
authorizes each step separately.

- **Step 1, database: done.** `retrieval_labs.db` (SQLAlchemy 2.1.4, psycopg 3.3.6, pgvector
  0.5.0): `documents`, `chunks` (`vector(384)`, nullable for BM25), `processing_jobs`
  (`job_state` enum: PENDING/RUNNING/COMPLETED/FAILED), `analysis_results` (JSONB report).
  Alembic 1.20.0 migration `0001`; a test compares the migrated schema with the models.
  `compose.yaml` runs `pgvector/pgvector:0.8.7-pg18` with no published port;
  `compose.dev.yaml` publishes it on `127.0.0.1:55432` only. Integration tests
  (`-m integration`) need `TEST_DATABASE_URL`, refuse database names not ending in `_test`,
  and run in a new CI job with a Postgres service container. `retrieval_labs.db` is in the
  leakage test's inference modules.
- **Step 2, persistence and synchronous processing: done.** `db/repository.py` (plain
  functions) and `worker/process.py::process_job`, which runs the unchanged
  `analyze_contract` on the stored text and saves chunks (offsets only; embeddings for
  dense methods) and the report. Claiming a job is a conditional `UPDATE ... WHERE state =
  'PENDING'`, so a duplicate delivery is skipped and never repeats an extractor call. No
  retries: any error ends in FAILED with a short `error_code` (`processing_error`,
  `budget_exceeded`, `document_corrupted`, `document_missing`); the traceback goes to the log
  only. The stored text hash is re-checked before processing. Integration tests cover each
  path; deliberately removing the PENDING guard or the hash check made the matching test fail.
- **Step 3, queue and worker: done.** RQ 2.12.0 + redis-py 8.1.0; `redis:8.10.2-alpine` in
  Compose (append-only persistence, no published port; `compose.dev.yaml` publishes
  `127.0.0.1:56379`). `worker/queue.py` enqueues with no retries and a time limit
  (`JOB_TIMEOUT_S`, default 900 s); `worker/main.py` (`python -m retrieval_labs.worker.main`)
  builds the retriever (`RETRIEVER`, default `hybrid+rerank`) and extractor (`LLM_PROVIDER`,
  default `fake`; `anthropic` refuses to start without `LLM_MAX_USD`) once and runs an RQ
  `SimpleWorker` (no fork per job; timer-based timeouts on Windows). Owner decision: a job
  left RUNNING by a dead worker becomes FAILED `worker_lost`, never re-queued (D-003);
  results of a job marked lost while still running are rolled back. Tests: queue → worker →
  COMPLETED, no retries, double enqueue runs once, lost-job sweep, a real 1 s timeout ending
  in `timeout` (on Windows), and a manual CLI worker run against the application database.
- **Step 4, API: done** (within its 3-day estimate). FastAPI 0.142.4 + uvicorn 0.54.0 +
  python-multipart 0.0.32 (`retrieval_labs.api`). Endpoints:
  - `GET /health` (503 `degraded` if the database or queue is down);
  - `POST /contracts` (multipart UTF-8 `.txt`, optional `title`; 202 with ids and URLs);
  - `GET /contracts/{id}/status` (also marks lost jobs);
  - `GET /contracts/{id}/report` (409 `report_not_ready` / `job_failed`);
  - `DELETE /contracts/{id}` (204; 409 while running; cascades to chunks, jobs, reports);
  - `GET /benchmarks/latest` (stored `phase4-test` rows via `report_tables`).

  Uploads: `MAX_UPLOAD_BYTES` (default 2 MB; oversized bodies refused before multipart
  parsing), `.txt` + `text/plain` only, strict UTF-8, no NUL, not blank. Text stored byte
  for byte (tested with CRLF and non-ASCII). Filenames are sanitized labels only, never
  paths. One error shape `{"error": {"code", "message"}}`; messages never come from
  exceptions, so no traceback or secret reaches clients (tested). An unreachable queue
  answers 503 and fails the job `queue_unavailable` within seconds (LOG-017). No auth or
  rate limiting: local only. Verified with a real uvicorn on `127.0.0.1:8000` (only
  listener), `curl` and a separate worker process: submit → COMPLETED → report → delete.
- **Step 5, configuration and logging: done.**
  - `retrieval_labs.config.Settings` (Pydantic) is the only place the API and worker read the
    environment; Alembic still reads `DATABASE_URL` alone.
  - Invalid settings stop the process with exit code 2, naming the variables without echoing
    values. The paid provider cannot be configured without `LLM_MAX_USD`.
  - `retrieval_labs.logging_setup` writes JSON lines on stdout for our loggers, uvicorn and
    RQ, with ids, hashes, sizes and durations as fields.
  - New entry points: `python -m retrieval_labs.api` (binds `API_HOST`, default 127.0.0.1;
    no `Server` header) and `python -m retrieval_labs.worker.main`.
  - `.env.example` lists every setting, checked by a test.
  - An end-to-end test (submit → worker → report) checks that every log line is JSON and
    that no contract text appears. Logging a slice of the text on purpose made both log
    tests fail.
  - Real processes: 9 API and 12 worker log lines, all JSON, no contract text.
- Next: step 6, Docker image and full Compose stack (asks owner approval before the model
  download).

**CI (brief 14.2): added before Phase 6.** `.github/workflows/ci.yml` runs on every push and
pull request, Linux only (owner decision; Windows is covered by local use): ruff lint and
format check, `pytest -m "not data and not smoke"`, and a separate `pytest -m smoke` step
(synthetic-fixture evaluation, not benchmark results). It downloads neither CUAD nor model
weights (`HF_HUB_OFFLINE=1`). uv cache is keyed on `uv.lock`. On Linux, torch now comes
from the PyTorch CPU index, which removes 19 CUDA packages (~2.4 GB) from the lock; the
Windows and macOS torch wheels are unchanged (LOG-016). Actions pinned by commit SHA,
releases at least two weeks old. CI badge at the top of the README.

**Minimum publishable version (brief 21.2, D-002): complete.** Waiting for the owner's demo
screenshot, the decision to make the repository public, and authorization to start Phase 5.

- Shared retriever factory; the pipeline defaults to `hybrid+rerank` (D-001). The dev
  benchmark re-run after the refactor reproduced every item file and run summary.
- `demo/streamlit_app.py`: top-5 passages per clause type for any CUAD test contract, presence
  threshold fit on train, benchmark tables rendered from stored runs. No LLM calls, no gold.
  Local only, telemetry off (`.streamlit/config.toml`, LOG-015). Retrieval with warm
  embeddings took 0.62–3.25 s per contract on three contracts (LOG-014).
- `streamlit==1.65.0` in a default `demo` dependency group; new packages are at least two
  weeks old (three transitive pins in `constraint-dependencies`).
- `README.md` (English) with a commented slot for a screenshot; `LICENSE` (MIT);
  `docs/DECISIONS.md` (D-001, D-002).
- Tests: report tables, demo (AppTest with and without data; import restrictions), README
  table identical to the stored test run.

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
| `uv run pytest` | 226 passed (includes 7 `data` tests and the hash-seed subprocess test) |
| `uv run ruff check .` / `ruff format --check .` | Clean |
| `uv run pytest -m "not data and not smoke"` / `-m smoke` | 211 passed / 8 passed (CI steps, run locally) |
| `run_benchmark.py` twice, `PYTHONHASHSEED` 11 and 97 | All outputs byte-identical (except run name) |
| `streamlit run demo/streamlit_app.py` (headless) | Health `ok`; only listener 127.0.0.1 |
| Phase 4 runs (dev, dev expanded, test) repeated with `PYTHONHASHSEED=97` | `summary.json`, item files, failure samples byte-identical; only timings/provenance differ |
| `run_pipeline.py --dev-first 3 --text-dir data/fixtures/adversarial --dry-run` | `--retriever bm25`: worst case US$ 0.0937; default `hybrid+rerank`: US$ 0.0935 |

Project LLM spend so far: US$ 0.00.

## Important technical decisions

1. Relevance criterion fixed before experiments; any-overlap and containment are sensitivity only.
2. Phase 3 used dev only; test was used once at the end of Phase 4 with five frozen methods.
3. Presence thresholds fit on train, reported on dev and test.
4. Benchmark and application share `ContractSearch`, so they cannot drift apart.
5. Queries frozen during Phase 3; failure analysis informs Phase 4.
6. Extraction path must stay runnable (CLAUDE.md); its tests passed unchanged after the refactor.
7. Linux installs use CPU-only torch (`[tool.uv.sources]`, LOG-016); a future GPU deployment
   would need to revisit this.

## Owner decisions (end of Phase 3)

1. **Chunking for Phase 4: 150/50.**
2. **Queries and extraction prompt stay frozen**, aligned with CUAD's written description, so
   Phase 4 measures whether dense/hybrid retrieval close the vocabulary gap BM25 cannot
   (LOG-010). After the main comparison, an optional "BM25 + expanded query" row, marked as
   tuned on dev and reported separately from the fair comparison.

## Pending tasks / next recommended action

1. Owner: capture a screenshot or GIF of the demo for the README slot; decide when to make
   the repository public; authorize Phase 5.
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
