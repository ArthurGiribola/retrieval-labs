# Status

_Last updated: 2026-10-20_

## Current phase

**Phase 2 — Functional MVP: in progress, paused before the paid pilot.** Steps 1–7 of the
Phase 2 plan are done (implementation and a free dry run). The paid pilot (step 8) waits for
the owner's authorization.

## Completed components

Phase 0 — Project Foundation: uv/Python 3.12 project, pytest, ruff, LF enforcement (LOG-001),
`scripts/inspect_cuad.py`.

Phase 1 — Dataset Preparation: `retrieval_labs.documents.Contract`; `retrieval_labs.cuad`
(download + SHA-256, parse, splits train 306 / dev 102 / test 102, stats, hash-verified store,
test-gold lock); `scripts/prepare_cuad.py` (byte-reproducible). LOG-002 to LOG-005.

Phase 2 — Functional MVP (so far):

- `clauses.py`: 5 clause types grouped into 4 report sections (renewal = term + notice).
- `ingestion/chunking.py`: word windows (150 words, 50 overlap) as exact source slices.
- `retrieval/`: shared interface; in-house BM25 with Lucene idf, k1 1.2, b 0.75 (LOG-006);
  one fixed query per clause type written from public CUAD category descriptions.
- `extraction/`: answer vs. result schemas; passage selection and quote verification
  (`extracted_text` is the contract's own slice, verbatim by construction); nonce-framed prompt;
  provider-neutral `Extractor`; `FakeExtractor`; `AnthropicExtractor` (Claude Haiku 5.5, effort
  medium, max_tokens 4000, cost-safe parsing, LOG-007); spend ledger with run cap and US$ 20
  project budget.
- `pipeline.py` + `scripts/run_pipeline.py` (dry run, fake and paid providers).
- `data/fixtures/adversarial/`: 4 prompt-injection cases with expectations pre-registered in
  `EXPECTED.md`.

## Tests executed

| Command | Result |
|---|---|
| `uv run pytest` | 140 passed (includes 6 `data` tests on the real prepared dataset) |
| `uv run ruff check .` / `ruff format --check .` | Clean |
| `run_pipeline.py --dev-first 1 --provider fake` | Real dev contract processed end to end (fake answers) |
| `run_pipeline.py --dev-first 3 --text-dir data/fixtures/adversarial --dry-run` | 35 calls planned; worst case US$ 0.0937 |

No paid API call has been made. Project LLM spend so far: US$ 0.00.

## Important technical decisions

1. BM25 implemented in-house with Lucene's always-positive idf (LOG-006).
2. 5 extraction calls per contract (one per CUAD-aligned clause type), 4 report sections.
3. Quotes verified against retrieved passages (exact, then whitespace/quote/dash tolerant);
   unverifiable FOUND is downgraded to UNCERTAIN.
4. `messages.create` + manual validation instead of `messages.parse`, so every billed call is
   recorded (LOG-007).
5. Inference modules may not import `retrieval_labs.cuad`; only `anthropic_extractor.py`
   imports the vendor SDK (both enforced by tests).
6. Chunks with zero BM25 score are never returned; a clause with no candidates is NOT_FOUND
   without a model call.

## Open issues for Phase 3

- **Long gold spans vs. chunk size (owner's observation).** With 150-word windows and 50-word
  overlap, only runs of up to 51 words are guaranteed to fit inside a single chunk
  (`tests/unit/test_chunking.py` documents the bound and its limit). Gold spans longer than
  that, common in `Cap On Liability` and `Renewal Term`, may straddle chunk boundaries. The
  Phase 3 relevance criterion must handle partial overlap explicitly (e.g. count a chunk as
  relevant if it overlaps a gold span by a defined fraction of the span or of the chunk), and
  extraction can only quote what the retrieved passages contain: a clause cut by a passage
  boundary can fail verification and end as UNCERTAIN (seen in a unit test with tiny chunks).
- Dev contracts are longer than test contracts (median 44.5k vs 25.7k chars, LOG-004).
- Verification proves a quote exists, not that it is a legitimate clause: an injected sentence
  quoted by the model would verify. The adversarial pilot measures this.

## Pending tasks / next recommended action

1. **Owner authorization for the paid pilot** (3 dev contracts + 4 adversarial cases, 35 calls,
   cap US$ 0.25). Requires `ANTHROPIC_API_KEY` in `.env`.
2. Analyze pilot outputs: statuses, verification rejections, adversarial results vs.
   `EXPECTED.md`, measured tokens vs. the worst-case bound.
3. Record findings (ENGINEERING_LOG), finish Phase 2 docs, stop for Phase 3 authorization.

## Known problems / risks

- Few answerable test contracts for some labels (16–29); per-label test metrics will be noisy.
- 13 overlapping gold span pairs; metrics must not assume disjoint spans.
- Near-duplicate contracts are not detected.
- LLM outputs are not deterministic (Haiku 5.5 rejects non-default sampling parameters); raw
  outputs and model settings are stored with every run.
- Playbook perspective for renewal notice (customer vs. vendor) undefined (Phase 5).
