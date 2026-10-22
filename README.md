# Retrieval Labs

[![CI](https://github.com/ArthurGiribola/retrieval-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurGiribola/retrieval-labs/actions/workflows/ci.yml)

Finding the evidence for contract clauses, and measuring how well different retrieval methods
do it. Retrieval Labs searches each contract in the
[CUAD](https://www.atticusprojectai.org/cuad) dataset for five clause types (governing law,
termination for convenience, cap on liability, renewal term and renewal notice period) and
compares four retrieval methods on held-out contracts, with confidence intervals.

> Technical demonstration, not legal advice. See [Limitations](#limitations).

## The problem

Contract review starts by finding the clause that answers a question: *which law governs
this agreement? can either party terminate without cause? is liability capped?* A language
model can only answer from the passages it is given, so the quality of the whole system is
bounded by retrieval. If the clause is not retrieved, the model either says nothing or, worse,
answers from the wrong text.

This project treats retrieval as the thing to measure:

- **Which retrieval method finds the right passage?** Lexical (BM25), semantic (dense
  embeddings), both fused, or fused and then reranked by a cross-encoder.
- **Can the retrieval score tell when a clause is missing?** Many contracts have no renewal
  or no cap on liability. Saying "insufficient evidence" must not be confused with proving
  the clause is absent.

## Results

Single run on the CUAD test split (102 contracts, 198 items where the clause is present),
after all methods were frozen on the dev split. Brackets are 95% confidence intervals from a
paired, contract-level bootstrap (1,000 resamples). ΔR@5 is the paired difference against
BM25. Latency is per query, on one CPU machine, with chunk embeddings cached.

| Method | R@1 | R@5 | MRR@10 | ΔR@5 vs BM25 | Presence AUROC | Query median (p95) |
|---|---|---|---|---|---:|---:|
| bm25 | 0.833 [0.766, 0.891] | 0.944 [0.906, 0.975] | 0.881 [0.828, 0.923] | — | 0.866 | 0.22 ms (1.35) |
| dense | 0.702 [0.637, 0.768] | 0.899 [0.848, 0.940] | 0.784 [0.734, 0.834] | -0.045 [-0.089, +0.000] | 0.818 | 16.92 ms (21.25) |
| hybrid | 0.813 [0.750, 0.871] | 0.955 [0.923, 0.979] | 0.869 [0.819, 0.912] | +0.010 [-0.006, +0.031] | 0.612 | 17.93 ms (23.39) |
| hybrid+rerank | 0.869 [0.815, 0.914] | 0.980 [0.959, 0.995] | 0.918 [0.883, 0.947] | +0.035 [+0.010, +0.067] | 0.919 | 431.30 ms (713.12) |
| bm25-expanded † | 0.879 [0.824, 0.929] | 0.975 [0.944, 0.995] | 0.919 [0.877, 0.953] | +0.030 [+0.009, +0.058] | 0.885 | 0.31 ms (1.79) |

† Not part of the fair comparison: its query was rewritten after studying dev-split failures.

- **R@k**: share of items where a relevant passage is among the top *k*. A passage is
  relevant when it overlaps at least half of the shorter of the passage and the annotated
  span (criterion fixed before any experiment).
- **MRR@10**: mean reciprocal rank of the first relevant passage.
- **Presence AUROC**: how well the top-1 score separates contracts that contain the clause
  from contracts that do not (macro average over the five clause types; thresholds fit on
  the train split).

**The application uses hybrid + reranking**: it is the best method of the fair comparison and
the best at detecting presence. At about 0.4 s per query, a contract takes roughly two
seconds for all five clause types, which is acceptable for review work.

Full results, per clause type and on dev: [docs/EVALUATION.md](docs/EVALUATION.md). Raw run
outputs: [reports/benchmark/](reports/benchmark/).

## Findings

**Fusion scores are not confidence scores** ([LOG-012](docs/ENGINEERING_LOG.md)). Hybrid
retrieval ranks as well as BM25 (ΔR@5 +0.010 on test, interval spanning zero), but its
presence AUROC falls to 0.612. Reciprocal rank fusion scores each passage by its ranks in the
component lists, so the top-1 score says where a passage ranked, not how well it matched: on
dev, 510 items produced only 31 distinct top-1 scores, and 172 of them tied at the maximum.
Reranking restores a content-based score, which is why hybrid + reranking detects presence
best. A good ranking metric is not a good confidence signal.

**A dev-tuned fix keeps about half its gain on unseen contracts**
([LOG-013](docs/ENGINEERING_LOG.md)). BM25 missed many cap-on-liability clauses because CUAD's
annotators also mark exclusions of damages and sole-remedy clauses, which the query, written
from CUAD's own label description, did not cover. An expanded query fixed this on dev
(ΔR@1 +0.084) but only +0.045 on test, the optimism expected from tuning on the evaluation
set. It was therefore kept out of the fair comparison. Meanwhile the cross-encoder reranker,
not significant on dev (ΔR@5 +0.025 [+0.000, +0.052]), was significant on test (+0.035
[+0.010, +0.067]).

**Dense retrieval alone is worse than BM25** on this task (ΔR@1 −0.131 [−0.200, −0.058] on
test). Its clearest gain is on cap on liability, the clause type with the vocabulary gap
(test R@1 0.682 against 0.614); it is level with or slightly ahead of BM25 on renewal term and behind on the other
three, most of all on the renewal notice period (R@1 0.250 against 0.812). One plausible
reason, not tested here, is that most clause types use stable legal wording that lexical
matching already captures.

## Architecture

```mermaid
flowchart LR
    A["CUAD raw JSON"] -->|"prepare_cuad.py<br/>SHA-256 verified"| B["Contracts<br/>train / dev / test"]
    A --> G[("Gold annotations")]
    B --> C["Chunking<br/>150 words, 50 overlap<br/>exact character offsets"]
    C --> D1["BM25"]
    C --> D2["Dense<br/>bge-small-en-v1.5"]
    D1 --> E["Reciprocal rank fusion"]
    D2 --> E
    E --> F["Cross-encoder rerank<br/>ms-marco-MiniLM-L6-v2, top 20"]
    F --> H["Top-5 passages + presence threshold"]
    H --> I["LLM extraction<br/>verbatim quotes verified<br/>implemented, not yet evaluated"]
    H --> J["Benchmark<br/>Recall@k, MRR, AUROC, bootstrap"]
    G --> J
```

- **One contract at a time.** Every index is built from the chunks of a single contract, so
  retrieval can never return text from another document.
- **Gold annotations stay in evaluation code.** Inference code receives a `Contract` with no
  annotation fields; a test fails if any inference module imports the gold module.
- **The benchmark and the application share one code path** (`ContractSearch` and the
  retriever factory), so what was measured is what runs.
- **Offsets are sacred.** Text is never normalized in a way that moves characters; the
  repository enforces LF line endings because Windows CRLF conversion would shift every
  offset ([LOG-001](docs/ENGINEERING_LOG.md)).
- **Extraction** (implemented, not yet evaluated): a nonce-framed prompt treats the contract
  as untrusted data, the model must quote the contract verbatim, and every quote is checked
  against the original text. A spend ledger caps every paid run.

The same pipeline also runs as a backend service: an API stores each uploaded contract and
queues a job, and a separate worker process runs the analysis and stores the result.

```mermaid
flowchart LR
    U["Client"] -->|"POST /contracts<br/>UTF-8 .txt"| API["FastAPI"]
    API -->|"document + PENDING job"| DB[("PostgreSQL + pgvector")]
    API -->|"job id"| Q[("Redis queue")]
    Q --> W["Worker<br/>same pipeline as the benchmark"]
    W -->|"chunks, embeddings,<br/>report, job state"| DB
    U -->|"GET status / report"| API
    API --> DB
```

```text
src/retrieval_labs/
  cuad/          dataset download, parsing, splits, hash-verified store, gold annotations
  ingestion/     chunking with character offsets
  retrieval/     BM25, dense, hybrid (RRF), reranker, factory, queries
  extraction/    prompt, Anthropic client, evidence verification, budget ledger
  evaluation/    relevance criterion, retrieval metrics, bootstrap, report tables
  pipeline.py    retrieval + extraction for one contract
  api/           FastAPI app, upload validation, error responses
  worker/        queue, job processing, worker entry point
  db/            SQLAlchemy models and repository functions
  config.py      settings read from the environment (the only place that reads it)
migrations/      Alembic migrations
scripts/         prepare_cuad.py, run_benchmark.py, run_pipeline.py,
                 download_models.py, api_smoke.py
demo/            Streamlit demo
docker/          image build check, Postgres init script
reports/         stored benchmark runs (summaries, per-item results, timings)
docs/            evaluation, dataset notes, decisions, engineering log
```

## Demo

![Retrieval Labs demo: top passages for each clause type in a CUAD test contract](docs/images/demo.png)

A Streamlit app shows, for any contract of the CUAD test split, the top five passages
retrieved for each clause type and whether the best one clears the presence threshold, next
to the benchmark tables. It makes no paid API calls. LLM extraction is left out until it has
been evaluated.

```bash
uv run streamlit run demo/streamlit_app.py
```

The first start downloads and loads the two models (about 230 MB on disk); each new contract
is then embedded once and cached under `data/cache/`.

## Backend: API and worker

Requirements: Docker with Compose. Everything runs locally; the API has no authentication,
so it is published on `127.0.0.1` only.

```bash
docker compose --profile setup run --rm models   # once: pinned model weights (~216 MiB)
docker compose up -d --build --wait              # api, worker, postgres, redis (+ migrations)

# End-to-end check: submit a contract, wait for its job, verify the report, delete it.
# Without --file it sends the text of a CUAD test contract (needs prepare_cuad.py first).
uv run python scripts/api_smoke.py --file my_contract.txt
```

| Endpoint | Purpose |
|---|---|
| `POST /contracts` | Upload a UTF-8 `.txt` contract (multipart, optional `title`); returns 202 with the document and job ids |
| `GET /contracts/{id}/status` | Job state: `PENDING`, `RUNNING`, `COMPLETED` or `FAILED` with an `error_code` |
| `GET /contracts/{id}/report` | The stored analysis (409 while the job is not finished, or if it failed) |
| `DELETE /contracts/{id}` | Remove the contract, its chunks, jobs and report (409 while running) |
| `GET /benchmarks/latest` | The stored test-split benchmark table |
| `GET /health` | Database and queue status (503 if either is down) |

Interactive documentation is served at `http://127.0.0.1:8000/docs`.

- **One image, several services.** The API, worker, migrations and model download share one
  non-root image (2.09 GB) with CPU-only torch; the build fails if a CUDA package is
  installed. Model weights live in a Docker volume, downloaded once by the only step that
  goes online, after checking the pinned revisions and file sizes. The API and worker run
  offline.
- **Jobs are never retried automatically** ([D-003](docs/DECISIONS.md)). A job may have made
  paid LLM calls before failing, so a retry must be someone's decision. Failures end in
  `FAILED` with a short code (`timeout`, `worker_lost`, `document_corrupted`, ...); tracebacks
  go to the worker log, never to clients.
- **Containers use the free extractor only** ([LOG-018](docs/ENGINEERING_LOG.md)). The spend
  ledger lives on the host, and a paid worker in a container would keep a second one, so the
  budget cap could be exceeded without either ledger noticing.
- **Uploads are validated before anything is stored**: size limit (2 MB by default),
  `.txt` only, strict UTF-8, no NUL bytes. The text is stored byte for byte, so evidence
  offsets point into exactly what was uploaded.
- **Logs are JSON lines on stdout** with ids, hashes, sizes and durations; they never contain
  contract text (tested).

Settings are read from the environment; [.env.example](.env.example) lists every one.

## Running it

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 from
`.python-version`). Tested on Windows 11 with CPU only.

```bash
uv sync                                              # dependencies, including the demo
uv run python scripts/prepare_cuad.py --download     # CUAD (~18 MB zip), verified by SHA-256
uv run pytest                                        # tests (dataset tests skip without data)

# Reproduce the dev comparison (writes reports/benchmark/<run-name>/)
uv run python scripts/run_benchmark.py --run-name my-dev-run \
    --method bm25 --method dense --method hybrid --method hybrid+rerank

# Run the pipeline without any LLM call: real prompts, worst-case cost estimate
uv run python scripts/run_pipeline.py --dev-first 3 --dry-run
```

Benchmark outputs are deterministic: running the same benchmark twice with different
`PYTHONHASHSEED` values produced byte-identical summaries and per-item results. The test
split requires `--final` and was used once.

Integration tests need PostgreSQL and Redis. They wipe their database, so they refuse any
database whose name does not end in `_test`:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --wait   # also publishes 127.0.0.1:55432 and :56379
TEST_DATABASE_URL=postgresql+psycopg://retrieval_labs:retrieval_labs@127.0.0.1:55432/retrieval_labs_test \
TEST_REDIS_URL=redis://127.0.0.1:56379/0 uv run pytest -m integration
```

CI (GitHub Actions, Linux) has three jobs: ruff lint and format checks, the unit tests that
need neither CUAD nor model weights and a deterministic evaluation smoke test on a synthetic
fixture (`pytest -m smoke`); the integration tests against PostgreSQL + pgvector and Redis
service containers; and a build of the Docker image. The smoke test checks that evaluation
runs; its numbers are not benchmark results. The full benchmark runs locally. On Linux,
torch comes from the CPU-only PyTorch index, so installs skip about 2.4 GB of CUDA packages.

## Status

| Component | Status |
|---|---|
| Dataset preparation, splits, leakage guards | Done |
| BM25, dense, hybrid and reranked retrieval | Done, evaluated on dev and once on test |
| Retrieval-level presence detection | Done, evaluated |
| Streamlit demo (retrieval only) | Done |
| LLM extraction with verbatim-quote verification | Implemented and unit-tested; **not yet evaluated** (the paid pilot has not run) |
| Backend: API, Redis queue, worker, PostgreSQL + pgvector, Docker Compose | Done; local only, no authentication |
| Playbook alerts | Planned |
| Web interface for uploads, deployment | Planned |

## Limitations

- **Not legal advice.** The system finds passages; it does not interpret them, and it cannot
  detect every risk in a contract.
- **"Insufficient evidence" is not absence.** It means the best passage scored below a
  threshold fit on the train split.
- **CUAD is a specific corpus**: 510 English-language commercial contracts filed with the
  US SEC. Results may not transfer to other jurisdictions, languages or contract types.
- **Relevance follows CUAD's annotations**, including their choices about what counts as a
  cap on liability ([LOG-009](docs/ENGINEERING_LOG.md)).
- **Small samples per clause type** (16 to 89 items per split): intervals are wide, and
  per-clause numbers should be read with care.
- **Dev and test are not directly comparable**: dev contracts are longer than test
  contracts, and every method scored higher on test.
- **Latency** comes from one development machine (CPU, single process).
- **The backend is for local use.** It has no authentication or rate limiting, accepts only
  UTF-8 `.txt` files, and runs a single worker.

## Data, models and license

- **Code:** [MIT License](LICENSE).
- **Data:** [Contract Understanding Atticus Dataset (CUAD) v1](https://www.atticusprojectai.org/cuad)
  by The Atticus Project, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is not redistributed
  here; `scripts/prepare_cuad.py` downloads it from the official repository. Hendrycks et al.,
  *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021.
- **Models:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT)
  and [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
  (Apache 2.0), pinned to exact revisions in the code.

Engineering decisions and the problems found along the way are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md) and [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).
