# CLAUDE.md — Retrieval Labs

Permanent conventions for working on this repository. The complete specification is
[docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md); **Section 21 (Addendum) overrides earlier
sections**. Current progress lives in [docs/STATUS.md](docs/STATUS.md); verified dataset facts
live in [docs/DATASET.md](docs/DATASET.md).

## Workflow

- Work one phase at a time (brief Section 16). Stop at the end of each phase and wait for the
  owner's authorization before starting the next.
- Each phase is time-boxed to 1–2 weeks; if it runs over, stop and propose a scope reduction.
- Do not change scope or technology choices silently; record decisions in `docs/STATUS.md`
  (and later `docs/DECISIONS.md`, once it has real content).
- Update `docs/STATUS.md` at the end of every phase.
- Maintain `docs/ENGINEERING_LOG.md`, an interview-oriented engineering diary. Add an entry
  (`LOG-NNN`, following the template in that file) whenever a bug, risk, technical decision or
  surprising result is **confirmed by code, data or command output** — never for speculation.
  Include decisions and surprising results, not only bugs. Entries are append-only.
- Ask before: paid API calls (estimate cost first; total LLM budget < US$ 20), downloads of large
  assets, cloud deployment, creating GitHub resources, or destructive operations.
- Explanations to the owner are in Portuguese; code, commits and documentation are in English.

## Environment

- Windows 11. Python 3.12 managed by **uv** (`.python-version`). Never use the system Python.
- Install/sync: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .` — Format: `uv run ruff format .`
- Add a dependency only in the phase that uses it: `uv add <pkg>` / `uv add --dev <pkg>`.

## Code conventions

- `src/` layout; package `retrieval_labs`. Type hints throughout; Pydantic for validated schemas.
- Prefer simple, explicit code over abstractions. No empty placeholder modules.
- Read JSON/text with explicit `encoding="utf-8"`; never normalize text in a way that changes
  character offsets of the original contract.
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`), scoped to the
  current phase.

## Data rules (non-negotiable)

- Datasets, model weights, `.env` and derived data are never committed (`data/raw/`,
  `data/processed/` are git-ignored).
- Read CUAD from the raw official JSON, not through the Hugging Face loader (it strips text
  without fixing offsets).
- Gold annotations are used **only** inside evaluation code. Inference code must never receive
  them. Keep separate types for source documents, indexes, gold annotations, predictions and
  evaluation results, and test that separation.
- Retrieval for an evaluation item searches only that item's own contract.
- Splits are contract-level. The official test split (102 contracts) is never used to tune
  chunking, retrieval parameters, prompts or thresholds — use the dev split derived from train.

## Honesty rules

- Never fabricate metrics, latencies, costs or completed features. Every published number must
  come from an executed run whose raw output is stored.
- Never claim a test passed without running it and inspecting the result.
- `extracted_text` must be a verbatim substring of the contract; interpretation goes in
  `normalized_value`. `NOT_FOUND` means "no sufficient evidence", not proof of absence.
