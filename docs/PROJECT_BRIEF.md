# RETRIEVAL LABS
## Master Project Brief — AI Engineering, Legal NLP & Retrieval Evaluation

**Project owner:** Arthur Giribola  
**Project name:** Retrieval Labs  
**Proposed repository:** `retrieval-labs`  
**Language:** Python 3.12  
**Primary focus:** AI Engineering / Backend Engineering / Information Retrieval / NLP  
**Project type:** Independent, portfolio-grade engineering project  
**Development approach:** Incremental development, benchmark-first, test-driven where practical.

> **Important:** Section 21 (Addendum) at the end of this document contains final adjustments that take priority over earlier sections.

---

# 0. Instructions to Claude Code

You are acting as a Senior AI Engineer, Software Architect and technical mentor.

Your responsibility is to help me design and implement Retrieval Labs from scratch.

This is a serious engineering portfolio project intended to demonstrate my technical abilities when applying for Junior AI Engineer, Python Backend Engineer and Applied ML Engineer positions.

Treat it like a real engineering project, not a coding tutorial or a quick AI-generated prototype.

I want clean architecture, reproducible experiments, measurable results, meaningful tests, technical documentation and a functional application.

### Mandatory development rules

1. Do not implement the entire project in one execution.
2. Read and understand this complete specification before proposing code.
3. Start with Phase 0 only.
4. Implement one phase at a time, following the acceptance criteria.
5. Do not silently change the project's scope or technology choices.
6. Explain major technical decisions and their trade-offs.
7. Prefer simple, understandable solutions over unnecessary abstractions.
8. Do not introduce extra frameworks merely to make the architecture appear sophisticated.
9. Use Python type hints, meaningful naming and modular code.
10. Never fabricate benchmark results, accuracy, latency, costs or completed features.
11. Every metric presented publicly must come from actual executed experiments.
12. Maintain a clear separation between development data, evaluation data and application data.
13. Never expose API keys or credentials.
14. Do not modify or reuse code, databases or infrastructure from my existing projects.
15. Before external publication, cloud deployment, paid API usage or destructive operations, request my approval.
16. At the end of every phase, summarize what was implemented, tested, remains incomplete and should happen next.

### Mentorship requirement

I am building this project to improve professionally, not just to generate code.

For each major component, explain:

- What problem it solves.
- Why the selected approach makes sense.
- Which alternatives exist.
- How it works internally.
- How I could explain it during a technical interview.

The goal is for me to understand and defend the complete system.

---

# 1. Project Overview

## 1.1 Project name

Retrieval Labs.

Subtitle:

**Evidence-Grounded Contract Review & Retrieval Benchmark**

One-line description:

An AI-powered contract analysis system that extracts critical clauses, detects deviations from configurable legal playbooks, and benchmarks different information retrieval strategies against expert-annotated data.

## 1.2 The problem

Consider a fictional small law firm with three attorneys and one paralegal.

Whenever a commercial agreement arrives, the paralegal manually reads the document looking for four important contractual topics:

1. Governing law.
2. Termination for convenience.
3. Limitation of liability.
4. Renewal and notice requirements.

After locating these provisions, the paralegal compares them against the firm's internal contracting preferences.

Those preferences are often scattered across documents or exist only as institutional knowledge.

This process is repetitive, difficult to standardize and susceptible to oversight.

## 1.3 Proposed solution

Retrieval Labs receives a commercial agreement and:

1. Processes the document.
2. Identifies passages relevant to the four contractual topics.
3. Extracts the relevant provisions.
4. Returns structured information.
5. Compares extracted information with a configurable contracting playbook.
6. Generates alerts for deviations or missing information.
7. Shows the original textual evidence supporting every finding.

Instead of initially reviewing an unorganized 30-page document, the attorney receives a structured report highlighting provisions requiring attention.

The system supports legal review; it does not replace legal professionals or certify that a contract is legally safe.

## 1.4 The technical research question

The project's central research question is:

**Which information retrieval strategy identifies relevant contractual clauses most reliably, and what are the accuracy, latency and operational trade-offs of each strategy?**

We will compare four retrieval approaches against the same evaluation dataset.

| Strategy | Description |
|---|---|
| BM25 | Classical lexical retrieval |
| Dense Retrieval | Semantic retrieval using embeddings |
| Hybrid Retrieval | Combined lexical and semantic retrieval |
| Hybrid + Reranking | Hybrid candidate retrieval followed by a second-stage ranking model |

The project must measure these approaches rather than assume the most complex one is the best.

---

# 2. Project Objectives

The project has three major objectives.

### Objective A — Build a functional application

Create a working contract analysis pipeline that can identify relevant contractual provisions and provide evidence-grounded results.

### Objective B — Perform reproducible evaluation

Use the CUAD dataset to create an objective retrieval and extraction benchmark.

Compare the different methods using consistent evaluation conditions.

### Objective C — Demonstrate production-oriented engineering

Build an independently executable system with:

- Python.
- FastAPI.
- PostgreSQL and pgvector.
- Background processing.
- Docker Compose.
- Automated tests.
- GitHub Actions.
- Structured logging.
- A simple Streamlit demonstration.

However, advanced infrastructure must not be added before the core pipeline works.

---

# 3. Functional Scope

The initial application supports four functional categories.

## 3.1 Governing Law

Identify the contractual provision establishing the applicable governing law.

Extract the relevant jurisdiction when possible.

Example fictional playbook:

Accepted governing laws:

- New York.
- Delaware.

Any other jurisdiction should be flagged for manual review.

If the jurisdiction cannot be determined, return an uncertainty or missing-information alert.

## 3.2 Termination for Convenience

Identify whether a party can terminate an agreement without cause, subject to applicable conditions.

Extract the relevant passage and structured information when possible.

Important: termination for convenience is narrower than all possible forms of contractual termination.

Do not claim that the application detects every possible termination mechanism.

## 3.3 Limitation of Liability

Identify provisions addressing contractual caps on liability.

Extract the relevant passage and, when possible:

- Whether a cap exists.
- Monetary amount or applicable formula.
- Relevant exceptions.
- Whether the information is ambiguous.

Absence of a detected cap should generate a manual-review alert, not an automatic legal conclusion.

## 3.4 Renewal

Identify renewal-related provisions.

Extract:

- Renewal term.
- Whether automatic renewal is explicitly established.
- Required notice to prevent renewal.
- Notice period in days, when reliably identifiable.

Example fictional playbook:

Automatic renewal requires at least 30 days of prior notice for cancellation.

This rule must be interpreted according to a clearly defined contracting perspective.

Do not automatically classify longer or shorter notice periods as universally favorable or unfavorable.

---

# 4. Dataset — CUAD

## 4.1 Official dataset

Use:

**CUAD — Contract Understanding Atticus Dataset, Version 1.**

Official sources:

https://www.atticusprojectai.org/cuad/

https://github.com/The-Atticus-Project/cuad

https://huggingface.co/datasets/theatticusproject/cuad

CUAD contains:

- 510 commercial contracts.
- More than 13,000 expert-supervised annotations.
- 41 contractual clause categories.
- Original contract documents and structured annotations.

The dataset is published under CC BY 4.0. Read its complete license, documentation and notices regarding the underlying contracts before redistributing material.

Provide the required attribution.

## 4.2 Selected labels

The four functional categories must map to these CUAD annotation labels:

| Product category | CUAD label |
|---|---|
| Governing Law | Governing Law |
| Termination | Termination For Convenience |
| Liability | Cap On Liability |
| Renewal | Renewal Term |
| Renewal notice | Notice Period To Terminate Renewal |

Before implementation, inspect the original dataset and verify the exact names, structure and availability of these labels.

Do not assume all labels are present in every contract.

Some annotations may contain multiple acceptable answer spans.

## 4.3 Data source rules

Prefer the official structured CUAD dataset for the benchmark.

Inspect the actual JSON format and validate how contracts, paragraphs, questions, answers and character offsets are represented.

Do not design the data loader around an assumed schema without inspecting a real sample.

For the initial MVP, use the existing contract text rather than beginning with PDF extraction.

PDF ingestion can be introduced later.

Preserve the original contract text and the mapping between extracted passages and their original locations.

## 4.4 Dataset organization

Create a reproducible dataset preparation process.

The system must support:

- Loading the selected CUAD records.
- Identifying unique contracts.
- Extracting the selected label annotations.
- Preserving multiple valid spans.
- Tracking clauses without annotated answers.
- Creating a manifest of dataset version, source and split membership.
- Validating the integrity of source offsets.

Use the official contract-level split when appropriate and available.

If a separate development split is needed, derive it from training data using a fixed random seed and group contracts by their original document identity.

Never divide chunks from the same contract between development and test sets.

Never use the final test set to tune retrieval parameters, prompts or thresholds.

## 4.5 Data leakage prevention

This is mandatory.

The retrieval system must never access gold answer annotations during normal inference.

Gold annotations are allowed only inside the evaluation pipeline.

Index documents using their actual content, without inserting answer labels or gold spans as retrieval hints.

For each evaluation item, retrieve passages from its corresponding contract, not from the entire dataset of unrelated agreements.

Use independent data objects and interfaces for:

1. Source documents.
2. Search indexes.
3. Ground-truth annotations.
4. Model predictions.
5. Evaluation results.

Include automated tests that verify dataset separation and prevent accidental answer leakage.

---

# 5. High-Level Architecture

The final application should follow this general flow:

    User uploads a contract
               |
               v
       Document Ingestion
               |
               v
       Text Normalization
               |
               v
       Chunking + Indexing
               |
               v
       Retrieval Pipeline
               |
       +-------+--------+
       |       |        |
      BM25   Dense    Hybrid
                        |
                        v
                 Optional Reranker
                        |
                        v
                 Evidence Selection
                        |
                        v
                 Clause Extraction
                        |
                        v
                 Structured Results
                        |
                        v
                 Playbook Evaluation
                        |
                        v
                 Alerts + Evidence
                        |
                        v
                 Review Interface

A separate benchmark engine evaluates retrieval and extraction using the CUAD reference annotations.

The application and benchmark must share the relevant retrieval implementation rather than maintain duplicate algorithms.

---

# 6. Technical Architecture

## 6.1 Python and project management

Use Python 3.12.

Prefer:

- `pyproject.toml` for project configuration.
- A reproducible dependency management approach.
- Type hints throughout the core application.
- Pydantic for request and response validation.
- Pytest for tests.
- Ruff for linting and formatting.

Choose a clear package structure without creating unnecessary architectural layers.

## 6.2 Document processing

Build an ingestion module responsible for:

- Reading source documents.
- Validating input.
- Preserving original text.
- Normalizing text carefully.
- Dividing documents into retrievable chunks.
- Maintaining reliable text offsets.
- Recording document and chunk identifiers.

Each chunk should contain at least:

- Unique chunk ID.
- Contract ID.
- Original text.
- Start character offset.
- End character offset.
- Chunking configuration metadata.

Do not normalize text in a way that silently invalidates CUAD's original annotation offsets.

Experiment with chunk size and overlap using the development set.

Do not select these values based on final test performance.

## 6.3 BM25 retrieval

Implement BM25 as the initial baseline.

A lightweight Python library such as `rank-bm25` is acceptable.

Use consistent preprocessing and query templates.

The module should return ranked retrieval results with:

- Chunk ID.
- Document ID.
- Score.
- Rank.
- Original text.
- Source offsets.

All retrieval methods must eventually follow the same result interface.

## 6.4 Dense retrieval

Implement semantic retrieval using a pretrained embedding model.

Sentence Transformers is an appropriate initial choice.

Choose a model based on:

- Suitability for English-language retrieval.
- Local execution requirements.
- Model size.
- Inference speed.
- Documentation and licensing.

Record the exact model name and version used in experiments.

During benchmark comparisons, hold the dense embedding model constant.

Use PostgreSQL with pgvector when the infrastructure phase is introduced.

Do not use the original annotations to influence embedding generation.

## 6.5 Hybrid retrieval

Implement a combination of BM25 and dense retrieval.

Reciprocal Rank Fusion (RRF) is an acceptable initial approach.

Expose the relevant parameters through configuration.

Avoid introducing an arbitrary weighting system without documenting its reasoning.

All parameters must be selected using development data.

## 6.6 Hybrid retrieval with reranking

Retrieve a larger candidate set using the hybrid method.

Then apply a pretrained cross-encoder reranking model.

Use the reranker to reorder candidates before selecting the final context.

Keep retrieval depth and final context budgets consistent when comparing the methods.

Record the reranker model name, version and execution environment.

Do not label reranking as inherently superior. Benchmark it.

---

# 7. Clause Extraction

The retrieval module identifies candidate passages.

The extraction module reads those passages and returns structured data.

Use a configurable LLM provider through an isolated interface.

Do not tightly couple application logic to a single proprietary SDK.

The initial provider may be selected according to available credentials, supported structured outputs, operating cost and reliability.

A deterministic test provider or mocks must be available so automated tests can run without paid API calls.

## 7.1 Extraction requirements

For every target category, the system must return one of three statuses:

- `FOUND`
- `NOT_FOUND`
- `UNCERTAIN`

Important distinction:

`NOT_FOUND` means the system found no sufficient evidence. It must not be presented as definitive proof that the clause does not exist anywhere in the contract.

## 7.2 Structured output

Define a Pydantic schema containing fields such as:

    clause_type
    status
    extracted_text
    normalized_value
    evidence
    extraction_notes

Evidence should include:

    document_id
    chunk_id
    source_start
    source_end
    exact_quote

The source offsets must refer back to the original document text.

If the system cannot verify that a quoted passage exists in the source document, it must not present that passage as verified evidence.

For values requiring interpretation, use explicitly typed optional fields where possible.

Examples:

- Governing-law jurisdiction.
- Automatic-renewal status.
- Notice period in days.
- Liability cap expression.

Do not invent information when a value cannot be determined reliably.

## 7.3 Prompt safety

Contracts are untrusted input.

Instructions embedded in a contract must be treated as document content, never as instructions to the application.

Test the extractor with adversarial passages that attempt to manipulate its behavior.

The extractor must not generate unsupported facts or pretend to have reviewed documents outside the supplied context.

---

# 8. Playbook and Alert Engine

Create a configurable contracting playbook.

Use YAML or JSON initially.

Example fictional rules:

    governing_law:
      accepted_jurisdictions:
        - New York
        - Delaware

    termination_for_convenience:
      requires_manual_review_if_missing: true

    liability:
      requires_cap: true

    renewal:
      minimum_notice_days: 30

This example is illustrative, not validated legal guidance.

## 8.1 Rule engine

Prefer deterministic Python logic for checking structured values against playbook rules.

Do not use an additional LLM to make every policy decision when ordinary programmatic rules are sufficient.

The system should distinguish between:

- A confirmed deviation from the fictional playbook.
- Missing evidence.
- Uncertain extraction.
- A provision that complies with the configured rule.

## 8.2 Alert structure

Each alert should include:

    alert_id
    clause_type
    severity
    status
    violated_rule
    description
    supporting_evidence
    review_required

Severity should be derived from documented playbook policies, not presented as universal legal risk.

When the data is incomplete, generate a manual-review notice instead of making unsupported legal conclusions.

## 8.3 Evaluation data for alerts

CUAD provides annotations for contractual provisions. It does not automatically provide the ground truth for this project's fictional playbook decisions.

Build a separate labeled evaluation set for the alert engine.

Target approximately 50 contracts or well-defined review cases, subject to practical availability and annotation quality.

Create explicit labeling guidelines.

Document:

- Which rule applies.
- Expected output.
- Evidence supporting the expected output.
- How ambiguity should be handled.

If legal professionals have not validated the labels, describe the benchmark as rule-based experimental evaluation rather than validated legal-risk detection.

---

# 9. Evaluation Framework

This is the most important technical component of Retrieval Labs.

Build a dedicated evaluation module that produces reproducible results.

Do not substitute subjective statements such as "the system works well" for objective measurements.

## 9.1 Retrieval metrics

Implement:

- Recall@1.
- Recall@3.
- Recall@5.
- MRR (Mean Reciprocal Rank).

Define the relevance criterion before running final experiments.

For example, document how overlap between a retrieved chunk and a gold annotation determines whether retrieval counts as successful.

Account for cases where a clause has multiple valid annotations or a gold span crosses chunk boundaries.

Report the selected relevance criterion clearly.

Compute answerable-case retrieval metrics separately from absence-detection results.

## 9.2 Extraction metrics

Implement:

- Exact Match where applicable.
- Token-level Precision.
- Token-level Recall.
- Token-level F1.
- Per-category results.
- Micro/macro averages where meaningful.

Handle multiple acceptable gold spans explicitly.

Compare extracted text with annotations using a documented normalization function.

Do not assume the most similar extracted passage is necessarily correct.

## 9.3 Presence and absence detection

Evaluate the system's ability to identify when evidence is missing.

Calculate classification metrics for the relevant labels.

Include confusion matrices when useful.

Distinguish failures of retrieval from failures of extraction.

## 9.4 Alert evaluation

Measure:

- Alert Precision.
- Alert Recall.
- False Positive Rate.
- False Negative Rate.
- Results by category.

Document the quality and origin of manually created labels.

## 9.5 Operational metrics

Record:

- Mean latency per contract.
- Median latency.
- P95 latency where sample size permits.
- Indexing time.
- Query/retrieval time.
- Reranking time.
- Extraction time.
- LLM token usage.
- Estimated LLM API cost.
- Model version.
- Hardware or deployment environment.
- Number of evaluated documents.

Distinguish query-time measurements from one-time indexing costs.

Do not invent monetary API costs for models executed locally. Report resource usage or estimated operational costs separately if appropriate.

## 9.6 Fair comparison

All four retrieval methods must use:

- The same evaluation contracts.
- The same target clause queries.
- A documented chunking strategy.
- The same final number of retrieved passages.
- The same extractor.
- The same extraction prompt.
- The same context budget.
- Consistent measurement conditions.

The reranking method may initially retrieve a larger candidate pool, but that difference must be explicitly documented.

Any differences in experimental conditions must be presented transparently.

## 9.7 Expected final report

Generate a table similar to:

| Method | Recall@1 | Recall@5 | MRR | Retrieval Latency |
|---|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |
| Hybrid + Reranking | TBD | TBD | TBD | TBD |

All values must come from actual experiments.

Store machine-readable raw results alongside the final Markdown report.

Record configurations and random seeds so experiments can be repeated.

A primary research finding may be that a simpler pipeline performs competitively with a more sophisticated one.

That is a valid result.

---

# 10. Backend API

The production-oriented backend should use FastAPI.

Initial proposed endpoints:

| Method | Endpoint | Responsibility |
|---|---|---|
| GET | `/health` | Application health |
| POST | `/contracts` | Submit contract for analysis |
| GET | `/contracts/{id}/status` | Processing status |
| GET | `/contracts/{id}/report` | Analysis results |
| GET | `/benchmarks/latest` | Latest completed benchmark summary |

Use appropriate HTTP status codes.

The asynchronous submission endpoint should return a job or document identifier and allow status polling.

## 10.1 Required behaviors

- Input validation.
- Structured error responses.
- Stable response schemas.
- Document identifiers.
- Configurable upload limits.
- Rejection of unsupported file types.
- Safe handling of failed jobs.
- Protection against malicious filenames.
- No stack traces or secrets exposed to clients.

Avoid publicly exposing an upload endpoint without suitable access controls, rate limits and operational safeguards.

---

# 11. Background Processing

Processing a contract may involve ingestion, embedding, retrieval and external LLM inference.

Long-running operations must not block API request handling.

For the production phase, introduce:

- Redis.
- RQ or another justified, maintained Python job processing solution.
- A separate worker process.

The API should submit processing jobs and return immediately with an identifier.

The worker should process the contract and persist results.

Implement clearly defined job states:

    PENDING
    RUNNING
    COMPLETED
    FAILED

Document retry behavior and error handling.

Avoid uncontrolled retries that could produce repeated paid API calls.

Keep job processing idempotent where practical.

Do not introduce queues or distributed processing during the earliest MVP phases.

---

# 12. Database

Use PostgreSQL.

Enable pgvector when vector retrieval is introduced.

Initial conceptual entities:

### documents
Stores document metadata and processing status.

### chunks
Stores indexed document passages, source offsets and embeddings.

### analysis_results
Stores completed extraction outputs and verified evidence.

### alerts
Stores playbook findings linked to analysis results.

### processing_jobs
Tracks background job state.

### benchmark_runs
Stores experiment configurations and summary results.

Do not introduce all tables during Phase 0.

The final schema should evolve according to actual implementation needs.

Use versioned database migrations.

The benchmark's gold annotations should remain logically separate from runtime application data.

---

# 13. User Interface

Use Streamlit.

The interface is a technical demonstration, not a full commercial SaaS platform.

Prioritize clarity, reliability and functionality over elaborate visual design.

## 13.1 Main screen

Include:

- Project name and short explanation.
- Contract upload.
- Analyze button.
- Processing status.
- Completed results.
- Error feedback.

## 13.2 Analysis results

Present four sections:

1. Governing Law.
2. Termination for Convenience.
3. Limitation of Liability.
4. Renewal.

Each section should display:

- Finding status.
- Extracted information.
- Exact supporting passage.
- Source reference.
- Applicable playbook alert.
- Review-required indicator.

## 13.3 Benchmark section

Display the comparison between retrieval methods.

Include results from actual executed experiments and clearly identify the dataset and evaluation configuration.

Do not display illustrative numbers as measured performance.

## 13.4 Visual direction

Professional, clean and minimal.

This is an AI engineering and legal document analysis tool.

Avoid a generic chatbot interface, excessive animations and unnecessary design complexity.

The interface should make the original evidence and evaluation results easy to inspect.

---

# 14. Infrastructure and Engineering Practices

The final project should support execution using Docker Compose.

Target services:

    api
    worker
    postgres
    redis
    demo

Not all services need to exist during the early phases.

Provide `.env.example` without real credentials.

Use environment variables for:

- Database configuration.
- Redis configuration.
- Model selection.
- LLM provider.
- API credentials.
- Retrieval configuration.
- Evaluation settings.

Implement structured application logging.

Do not log complete confidential contracts, credentials or unnecessary sensitive contents.

## 14.1 Testing

Create tests for:

- Data loading.
- Dataset integrity.
- Text chunking.
- Source offset preservation.
- BM25 retrieval.
- Dense retrieval.
- Hybrid combination.
- Reranking.
- Retrieval metric calculations.
- Extraction schema validation.
- Evidence verification.
- Playbook rules.
- API error handling.
- Background job behavior.

Use fixtures and mocks for external services.

The main test suite must not require paid LLM API calls.

## 14.2 Continuous Integration

Configure GitHub Actions to run:

- Linting.
- Formatting checks.
- Unit tests.
- A small deterministic evaluation smoke test.

The CI pipeline should not download and process the entire CUAD dataset on every commit.

Create a separate workflow or local command for the complete benchmark.

Clearly distinguish smoke-test results from full-benchmark results.

---

# 15. Proposed Repository Structure

This is the target structure, not a requirement to create every directory immediately.

    retrieval-labs/
    |
    |-- README.md
    |-- CLAUDE.md
    |-- pyproject.toml
    |-- .env.example
    |-- .gitignore
    |-- docker-compose.yml
    |
    |-- src/
    |   |-- retrieval_labs/
    |       |-- config.py
    |       |-- ingestion/
    |       |-- retrieval/
    |       |   |-- base.py
    |       |   |-- bm25.py
    |       |   |-- dense.py
    |       |   |-- hybrid.py
    |       |   |-- reranker.py
    |       |-- extraction/
    |       |-- evaluation/
    |       |-- playbook/
    |       |-- storage/
    |       |-- api/
    |       |-- workers/
    |
    |-- scripts/
    |   |-- prepare_cuad.py
    |   |-- run_pipeline.py
    |   |-- run_benchmark.py
    |
    |-- configs/
    |   |-- retrieval.yaml
    |   |-- playbook.yaml
    |
    |-- data/
    |   |-- README.md
    |   |-- raw/
    |   |-- processed/
    |   |-- fixtures/
    |
    |-- tests/
    |   |-- unit/
    |   |-- integration/
    |   |-- evaluation/
    |
    |-- demo/
    |   |-- streamlit_app.py
    |
    |-- reports/
    |   |-- benchmark/
    |
    |-- docs/
    |   |-- PROJECT_BRIEF.md
    |   |-- ARCHITECTURE.md
    |   |-- EVALUATION.md
    |   |-- DATASET.md
    |   |-- DECISIONS.md
    |   |-- ROADMAP.md
    |   |-- STATUS.md
    |
    |-- .github/
        |-- workflows/
            |-- ci.yml

Adapt this structure when a simpler implementation is technically justified.

Avoid empty placeholder modules with no useful purpose.

Do not commit downloaded datasets, model weights, database dumps, temporary files, virtual environments or credentials.

Include a reproducible dataset download/preparation procedure.

---

# 16. Development Roadmap

Follow these phases in order.

Each phase must end with working, testable deliverables.

## PHASE 0 — Project Foundation

Goal: establish the technical foundation and understand the dataset before coding the application.

Tasks:

1. Inspect my local development environment.
2. Verify Python and Git availability.
3. Inspect the official CUAD sources.
4. Confirm the actual dataset schema.
5. Confirm the target labels.
6. Analyze dataset sizes and initial requirements.
7. Propose the minimal initial architecture.
8. Define the development tools.
9. Create the initial project structure.
10. Configure dependency management, linting and Pytest.
11. Write `CLAUDE.md` with project-specific instructions.
12. Create initial technical documentation.

Deliverables:

- Working Python environment.
- Initial repository structure.
- Project documentation.
- Dataset inspection summary.
- First automated smoke test.
- Clear next-phase instructions.

Acceptance criteria:

The environment works, the chosen CUAD format is understood, selected labels are verified, and the project can execute its initial test successfully.

Do not build the retrieval pipeline yet.

## PHASE 1 — Dataset Preparation

Goal: transform the CUAD dataset into a clean, reproducible evaluation resource.

Tasks:

1. Implement the dataset download/verification procedure.
2. Parse the source annotation files.
3. Extract the selected categories.
4. Preserve contract IDs and answer offsets.
5. Identify multiple answer spans.
6. Establish contract-level dataset splits.
7. Validate source text integrity.
8. Generate dataset statistics.
9. Create automated integrity tests.

Acceptance criteria:

The prepared dataset can be regenerated, annotations map correctly to the source contracts, and the final evaluation set is protected from accidental leakage.

## PHASE 2 — Functional MVP

Goal: create the first end-to-end working analysis pipeline.

Use only BM25 initially.

Required flow:

    Contract text
        |
        v
    Chunking
        |
        v
    BM25 retrieval
        |
        v
    Evidence selection
        |
        v
    LLM extraction
        |
        v
    Structured JSON output

Use the command line.

No frontend, database or queue is required at this stage.

Acceptance criteria:

A real CUAD contract can be processed end-to-end, producing structured results for the target categories with verified textual evidence.

The pipeline must handle missing or uncertain clauses.

This is the first major functional milestone.

## PHASE 3 — Initial Evaluation

Goal: objectively evaluate the MVP.

Tasks:

1. Implement retrieval evaluation.
2. Implement extraction metrics.
3. Handle multiple gold spans.
4. Measure presence/absence performance.
5. Generate an initial reproducible report.
6. Document failure cases.

Acceptance criteria:

The BM25-based pipeline can be evaluated against CUAD annotations without using those annotations during inference.

Results are reproducible.

## PHASE 4 — Retrieval Comparison

Goal: implement and compare all four retrieval methods.

Add:

- Dense retrieval.
- Hybrid retrieval.
- Hybrid retrieval with reranking.

Use a shared retrieval interface.

Measure retrieval quality, latency and relevant operational costs.

Tune configurations using development data only.

Acceptance criteria:

The four methods can run against the same benchmark, and the system generates a documented comparison table.

## PHASE 5 — Playbook and Alerts

Goal: introduce configurable contracting rules and findings.

Tasks:

1. Create the playbook configuration.
2. Implement deterministic rule evaluation.
3. Produce structured alerts.
4. Add manual-review states.
5. Build the separate alert evaluation dataset.
6. Evaluate alert quality.

Acceptance criteria:

The system compares extracted provisions against a configurable playbook and provides traceable explanations without inventing legal conclusions.

## PHASE 6 — Backend and Infrastructure

Goal: make the system executable as an independent backend application.

Tasks:

- FastAPI endpoints.
- PostgreSQL.
- pgvector.
- Database migrations.
- Background job processing.
- Redis.
- Worker process.
- Docker Compose.
- API tests.
- CI pipeline.
- Structured logging.

Acceptance criteria:

The complete application can process a contract through the API using an independently running worker, with persisted status and results.

## PHASE 7 — Interface and Deployment

Goal: provide a functional public-facing technical demonstration.

Tasks:

- Streamlit interface.
- API integration.
- Contract analysis results.
- Evidence display.
- Benchmark dashboard.
- Error handling.
- Deployment configuration.

Use only authorized public or appropriately licensed example contracts in the public demonstration.

Do not accept arbitrary confidential third-party contracts on a public deployment without appropriate privacy and security controls.

Acceptance criteria:

A reviewer can understand and demonstrate the core project without reading the entire source code first.

Deployment is subject to my approval and available budget.

## PHASE 8 — Final Portfolio Delivery

Goal: transform the completed implementation into a strong engineering portfolio case study.

Required deliverables:

1. Professional README in English.
2. Architecture documentation and diagram.
3. Reproducible evaluation report.
4. Three-minute demonstration video or detailed recording plan.
5. Technical case study explaining major decisions.

The case study should address:

- What problem was solved?
- Why was CUAD selected?
- Why compare four retrieval methods?
- What were the results?
- Which approach performed best under the chosen criteria?
- What were the principal failure cases?
- What engineering trade-offs were encountered?
- What would improve with more time or data?

Never present unfinished functionality as completed.

---

# 17. Git and Documentation Standards

Use meaningful commits.

Prefer Conventional Commits:

    feat:
    fix:
    test:
    docs:
    refactor:
    chore:

Keep changes scoped to the current phase.

Maintain `docs/STATUS.md` throughout development.

At minimum, STATUS.md should document:

- Current phase.
- Completed components.
- Pending tasks.
- Known problems.
- Tests executed.
- Important technical decisions.
- Next recommended action.

`CLAUDE.md` should contain the permanent development conventions and explain where to find the complete project specification.

The project must remain understandable if development resumes in a new Claude Code session.

Do not claim a test passed unless it was actually executed and its result was inspected.

---

# 18. Security, Privacy and Limitations

Retrieval Labs is a technical demonstration and experimental contract-review system.

It is not a legal advisory service.

Requirements:

- Do not provide definitive legal advice.
- Do not claim all contract risks are detectable.
- Do not claim CUAD represents every jurisdiction or commercial agreement type.
- Treat uploaded contracts as untrusted material.
- Prevent document-based prompt injection.
- Protect API credentials.
- Validate uploaded files.
- Avoid retaining confidential content unnecessarily.
- Document external LLM data-processing implications.
- Verify source quotations.
- Clearly distinguish missing evidence from proven clause absence.
- Document limitations of the dataset and evaluation methodology.
- Respect CUAD attribution and applicable usage terms.

Explain these limitations in the README and demonstration interface.

---

# 19. Project Success Criteria

Retrieval Labs is considered complete when:

- The four contractual categories are supported.
- The data preparation pipeline is reproducible.
- All four retrieval methods are implemented.
- Retrieval and extraction metrics are calculated from real evaluation data.
- Gold annotations remain isolated from inference.
- Actual benchmark results are published.
- The playbook engine produces traceable findings.
- The application can handle missing and uncertain evidence.
- FastAPI and background processing work.
- Docker Compose starts the required services.
- Automated tests and CI pass.
- A usable demonstration exists.
- Documentation enables another developer to reproduce the project.
- Every published technical claim is supported by code, experiments or clearly identified documentation.

The goal is not to build the largest system possible.

The goal is to build a credible, technically defensible AI Engineering project.

---

# 20. FIRST EXECUTION — IMPORTANT

You now have the complete specification.

**Do not implement all phases. Start with Phase 0 only.**

Your immediate tasks are:

1. Read and understand this specification, including the Addendum in Section 21.
2. Inspect my actual local environment and available tools.
3. Check the official CUAD dataset documentation and structure.
4. Identify technical ambiguities or risks that must be resolved before implementation.
5. Propose the minimal project foundation.
6. Create the initial development environment and documentation.
7. Explain each initial file and dependency.
8. Execute the initial smoke test.

Do not download unnecessarily large assets, use paid APIs, deploy services or create external GitHub resources without confirmation.

Before writing substantial code, give me:

- A short technical summary of the project.
- The proposed initial architecture.
- The dependencies required for Phase 0.
- The planned directory structure.
- Any important assumptions that need confirmation.
- The precise Phase 0 implementation sequence.

Then proceed with Phase 0.

When Phase 0 is finished, show:

- Files created.
- Technical decisions.
- Tests executed and actual results.
- Remaining risks.
- What Phase 1 will implement.

**STOP and wait for my authorization before beginning Phase 1.**

The objective is to build Retrieval Labs correctly, incrementally and with full understanding of the underlying engineering decisions.

---

# 21. ADDENDUM — Final Adjustments (these take priority over earlier sections)

## 21.1 Environment
Use uv to manage the Python version and dependencies. The development
machine runs Windows 11, with Python 3.12 already installed via uv and
Git available. The system Python (3.14) must not be used for this project.

## 21.2 Minimum publishable version
After Phase 4, plus a simple Streamlit demo and an English README, the
project is publishable. Phases 5–7 are improvements on top of a working,
public project, not prerequisites for publishing it.

## 21.3 Time boxes
Each phase should take one to two weeks. If a phase runs past that,
stop and tell me, so we can reduce its scope instead of letting it drag.

## 21.4 LLM budget
Retrieval metrics (Recall@k, MRR) require no LLM calls; run them for all
four methods on the full evaluation set. Run LLM extraction only on a
subset of contracts, or only on the one or two best retrieval methods.
Before any run that calls a paid API, estimate the cost and ask for my
approval. Total LLM spend for the project must stay under US$ 20.

## 21.5 Verbatim extraction
The `extracted_text` field must be a verbatim copy of text from the
contract, never a paraphrase, because extraction metrics compare it
token by token with CUAD annotations. Any interpretation belongs in
`normalized_value`.

## 21.6 Ceiling effect
Because retrieval runs within a single contract, all methods may score
similarly on Recall@5. Investigate this in Phase 3: focus on Recall@1
and MRR, and test whether smaller chunks make the comparison more
informative. A small difference between methods is a valid finding.

## 21.7 Alert evaluation size
Start with 20–30 well-labeled alert cases. Expand toward 50 only if
time allows.

## 21.8 Documentation in Phase 0
Create only CLAUDE.md, docs/STATUS.md and docs/DATASET.md in Phase 0.
Add the other docs files when they have real content. This brief itself
may be moved to docs/PROJECT_BRIEF.md.