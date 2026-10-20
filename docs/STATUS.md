# Status

_Last updated: 2026-10-20_

## Current phase

**Phase 0 — Project Foundation: complete.** Waiting for authorization to start Phase 1.

## Completed components

- Git repository (local only, branch `main`, no remote).
- uv project pinned to Python 3.12 (`.python-version`, `pyproject.toml`, `uv.lock`), `src/` layout.
- Dev tooling: pytest and ruff, configured in `pyproject.toml`.
- Smoke test (`tests/test_smoke.py`).
- CUAD v1 `data.zip` downloaded locally and verified (SHA-256 in `docs/DATASET.md`).
- `scripts/inspect_cuad.py`: reproducible inspection of schema, splits, labels and offsets.
- Documentation: `CLAUDE.md`, `docs/DATASET.md`, this file. Brief moved to `docs/PROJECT_BRIEF.md`.

## Tests executed

| Command | Result |
|---|---|
| `uv run pytest` | 2 passed |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | All files formatted |
| `uv run python scripts/inspect_cuad.py data/raw/cuad/data.zip` | 13,823/13,823 offsets valid; 0 train/test overlap |

## Important technical decisions

1. **Data source:** official GitHub `data.zip`, because it contains the official contract-level
   train (408) / test (102) split.
2. **Raw JSON, no Hugging Face loader:** the loader strips text without adjusting offsets.
3. **Label names:** the brief's names (Title Case) match the JSON exactly; the GitHub
   `category_descriptions.csv` uses different capitalization and is not authoritative.
4. **Annotations:** the train file splits multi-answer questions into separate entries; merged by
   label they equal `CUADv1.json`. Phase 1 can therefore take annotations from `CUADv1.json` and
   split membership from the two split files (to be finalized in Phase 1).
5. **Minimal tooling:** no mypy for now (owner's decision); CI deferred until the repository is
   published on GitHub.

## Pending tasks / next recommended action

Phase 1 — Dataset Preparation (awaiting authorization):

1. Automated download with SHA-256 verification.
2. Typed loader for contracts and gold annotations (separate types).
3. Extraction of the five target labels with multiple spans and empty answers.
4. Contract-level dev split derived from train with a fixed seed; manifest file.
5. Integrity tests (offsets, split disjointness, no gold data in inference types) using small
   fixtures so tests do not need the full dataset.

## Known problems / risks

- Few answerable test contracts for some labels (16–29); per-label test metrics will be noisy.
- Contracts up to ~338k characters; chunking and LLM context costs need care.
- Playbook perspective for renewal notice (customer vs. vendor) is still undefined (Phase 5).
- LLM provider not chosen yet (Phase 2; cost estimate required before any paid call).
