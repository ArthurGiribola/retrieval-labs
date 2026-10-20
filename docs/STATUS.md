# Status

_Last updated: 2026-10-20_

## Current phase

**Phase 1 — Dataset Preparation: complete.** Waiting for authorization to start Phase 2.

## Completed components

Phase 0 — Project Foundation:

- Git repository (local only, branch `main`, no remote); `.gitattributes` enforcing LF (LOG-001).
  `docs/PROJECT_BRIEF.md` working copy normalized from CRLF to LF (content unchanged).
- uv project pinned to Python 3.12, `src/` layout, pytest and ruff.
- `scripts/inspect_cuad.py`: independent inspection of the raw archive.

Phase 1 — Dataset Preparation:

- `retrieval_labs.documents.Contract`: inference-safe source document (no annotation fields).
- `retrieval_labs.cuad`: `labels`, `gold` (evaluation only), `source` (download + SHA-256),
  `parse`, `splits` (seeded dev split by document identity), `stats`, `store` (JSONL + manifest,
  hash-verified loaders, test-gold lock), `prepare` (orchestration).
- `scripts/prepare_cuad.py`: CLI producing `data/processed/cuad/` (510 contracts,
  2,550 gold annotations, stats, manifest). Byte-reproducible across runs.
- Splits: train 306 / dev 102 (seed 42) / test 102 (official).
- Leakage guards: AST import boundary for `cuad.gold`, `Contract` field whitelist, contract file
  key check, test-gold lock.
- Docs: `docs/DATASET.md` (prepared dataset section), `docs/ENGINEERING_LOG.md` (LOG-003 to
  LOG-005), `CLAUDE.md` (data-loading rules).

## Tests executed

| Command | Result |
|---|---|
| `uv run pytest` | 65 passed (59 self-contained + 6 `data` tests on the real prepared dataset) |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | All files formatted |
| `uv run python scripts/prepare_cuad.py` (run twice) | Byte-identical outputs (SHA-256 compared) |
| U+2028 regression tests with the fix reverted | 5 failed (fix confirmed necessary) |

## Important technical decisions

1. Text and annotations from `CUADv1.json`; official split files used only for membership.
2. Contracts and gold annotations in separate types, modules and files.
3. Dev split: 102 contracts, seed 42, sampled by text-hash groups so identical documents share
   a split (LOG-003). Seed kept despite a measured dev/test shift (LOG-004).
4. Prepared files are pure ASCII JSON with LF; JSONL split on `\n` only (LOG-005).
5. No timestamps in outputs, so reproducibility can be checked by hash.
6. Phase 0 decisions still apply: official `data.zip`, raw JSON (no Hugging Face loader), label
   names as in the JSON, no mypy, CI deferred until the repository is published.

## Pending tasks / next recommended action

Phase 2 — Functional MVP (awaiting authorization): chunking with source offsets, BM25 retrieval,
evidence selection, LLM extraction with verbatim `extracted_text`, structured JSON output, CLI.
Decisions needed at the start of Phase 2: LLM provider and model (with cost estimate before any
paid call), chunking strategy, query templates per label.

## Known problems / risks

- Dev contracts are longer than test contracts (median 44.5k vs 25.7k chars); dev and test
  results are not directly comparable (LOG-004).
- Few answerable test contracts for some labels (16–29); per-label test metrics will be noisy.
- 13 overlapping gold span pairs; metrics must not assume disjoint spans.
- Near-duplicate contracts (non-identical text) are not detected.
- Contracts up to ~338k characters; chunking and LLM context costs need care.
- Playbook perspective for renewal notice (customer vs. vendor) undefined (Phase 5).
