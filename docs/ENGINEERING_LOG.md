# Engineering Log

A record of problems, decisions and surprising results, written to be explained in technical
interviews. Rules:

- Only facts confirmed by code, data or command output. Reasoning that was not verified is
  labeled as such.
- Decisions and surprising results belong here too, not only bugs.
- Entries are append-only; corrections are made in a new entry that references the old one.

Entry template:

```markdown
## LOG-NNN — <short title>

- **Date / Phase:** YYYY-MM-DD / Phase N
- **Category:** bug | risk | decision | surprising result | data quality | performance
- **Context:** what we were doing.
- **The problem:** what was wrong or at stake.
- **How it was found:** the command, test or inspection that revealed it.
- **Impact if missed:** concrete consequence.
- **Fix / decision:** what was done and why.
- **Evidence:** commands, files, numbers, commits.
- **Interview version:** 2–3 sentences.
```

---

## LOG-001 — CRLF conversion on Windows can shift character offsets

- **Date / Phase:** 2026-10-20 / Phase 0
- **Category:** risk / decision
- **Context:** Setting up the repository on Windows 11. The project's core invariant is that
  every evidence span is identified by character offsets into the original contract text.
- **The problem:** Git for Windows ships with `core.autocrlf=true` at the *system* level, so
  text files are checked out with CRLF line endings on Windows and LF on Linux. A CRLF file has one
  extra character per line break, so every offset after the first line moves.
- **How it was found:** `git commit` printed "LF will be replaced by CRLF" for every file.
  `git config --show-origin --get-all core.autocrlf` showed the value comes from
  `C:/Program Files/Git/etc/gitconfig`. `git ls-files --eol` showed `docs/PROJECT_BRIEF.md` with
  `i/lf w/crlf` (LF in the repository, CRLF on disk), which gave a real file to measure.
- **Impact if missed:** Measured on that file: the heading `# 21. ADDENDUM` sits at offset
  40,015 in the LF version and 41,446 in the CRLF version — a shift of 1,431, exactly the number
  of preceding lines. Whether this bites depends on how a file is read: Python's default text
  mode translates CRLF back to LF (offset 40,015), while `open(..., newline="")` or reading bytes
  keeps CRLF (offset 41,446). Plain-text contract fixtures read without newline translation
  would produce offset tests that pass on Linux CI and fail on Windows, or the reverse. The CUAD
  JSON itself is not affected: `CUADv1.json` contains 0 raw line breaks (all 197,813 newlines are
  escaped `\n` inside strings).
- **Fix / decision:** Added `.gitattributes` with `* text=auto eol=lf`, so every text file
  is LF in the working tree on every OS regardless of the machine's Git configuration. This
  removes the dependency on how each reader handles newlines, instead of relying on every
  future reader to be written carefully.
- **Evidence:** commit `4c56888` (`chore: enforce LF line endings to preserve text offsets`);
  `git ls-files --eol` shows `i/lf w/lf` for all files written in Phase 0. `PROJECT_BRIEF.md`
  still shows `w/crlf` because it was not re-checked out; the attribute applies on next checkout.
- **Interview version:** "My system extracts evidence by character offsets, and I noticed Git
  for Windows converts line endings to CRLF on checkout. I measured it on a real file: one heading
  moved by 1,431 characters, exactly one per preceding line. I pinned LF with `.gitattributes`
  so fixtures are byte-identical on Windows and Linux and offset tests can't diverge between my
  machine and CI."

## LOG-002 — Read raw CUAD JSON instead of the Hugging Face loader; all 13,823 offsets validated

- **Date / Phase:** 2026-10-20 / Phase 0
- **Category:** decision / data quality
- **Context:** Choosing how to load CUAD v1. The benchmark compares retrieved chunks against
  gold spans by character offset, so offsets must refer exactly to the original contract text.
- **The problem:** The official Hugging Face loader script (`theatticusproject/cuad-qa`,
  `cuad-qa.py`) calls `.strip()` on each `context` and each answer `text` but keeps
  `answer_start` unchanged. If any context had leading whitespace, or any answer had leading
  whitespace, the offsets would point to the wrong characters.
- **How it was found:** Reading the loader source before depending on it, then measuring the
  real data with `scripts/inspect_cuad.py`, which checks
  `context[answer_start : answer_start + len(text)] == text` for every answer span.
- **Impact if missed:** Had the data contained edge whitespace, gold spans would be silently
  misaligned, and every retrieval and extraction metric built on them would be wrong without any
  error being raised.
- **Fix / decision:** Load the official `data.zip` JSON directly with the standard library. The
  measurement showed the loader's `.strip()` is harmless on CUAD v1 today (0 contexts and 0
  answers with edge whitespace), but reading raw JSON guarantees offset correctness by construction
  rather than by coincidence, and keeps it true if the data or loader changes. The same inspection
  confirmed the official split is clean (408 train / 102 test, 0 contracts in both) and that
  merging the train file's "separate question" entries reproduces `CUADv1.json` exactly.
- **Evidence:** `uv run python scripts/inspect_cuad.py data/raw/cuad/data.zip` →
  train 11,180/11,180, test 2,643/2,643, full 13,823/13,823 offsets exact; 0 mismatches;
  `contexts_with_edge_whitespace: 0`, `answers_with_edge_whitespace: 0`. Archive SHA-256
  `f8161d18…b999a` (full value in `docs/DATASET.md`). Commit `3ece1f9`.
- **Interview version:** "Before building on CUAD, I read the official loader and saw it strips
  text without adjusting answer offsets. I wrote an inspection script that validated all 13,823
  gold spans against the raw text — every one matched, so the bug was latent, not active. I still
  load the raw JSON, because offset correctness is the foundation of every metric in the
  benchmark and I wanted it guaranteed, not lucky."

## LOG-003 — Two CUAD contracts share identical text but not identical labels

- **Date / Phase:** 2026-10-20 / Phase 1
- **Category:** data quality / decision
- **Context:** Designing the dev split, which is carved from the official train split. The
  brief requires grouping contracts by original document identity so a document cannot appear
  on both sides of a split.
- **The problem:** CUAD lists 510 contracts but has only 509 distinct texts.
  `ADUROBIOTECH,INC_06_02_2020-EX-10.7-CONSULTING AGREEMENT` and the same title with a `(1)`
  suffix are byte-identical (same SHA-256). Both are in the official train split, so the
  official train/test split is unaffected — but a random dev draw could put one copy in dev
  and the other in train. The two copies are also labeled differently: their annotations
  differ on 2 of 41 labels (`Exclusivity` and `Anti-Assignment` each have one span in one copy
  and none in the other). On the five target labels they are identical.
- **How it was found:** Hashing every contract text and grouping by hash before writing the
  split code, instead of assuming titles identify documents; then diffing the two copies'
  annotations label by label.
- **Impact if missed:** A train/dev leak of one document. Small here, but the same blind spot
  would let a near-copy reach the test set in another dataset. The label disagreement is also a
  measured, concrete example of annotation noise in the gold standard.
- **Fix / decision:** Keep both contracts (they are official records) and sample the dev split
  by *text-hash groups*, so identical texts always land in the same split. Identical text across
  the official train/test split raises an error instead of being silently resolved. The groups
  are recorded in `manifest.json` (`duplicate_text_groups`). Near-duplicates (non-identical
  text) are not detected; that is a documented limitation.
- **Evidence:** `cuad_8c7e7f51de1f` and `cuad_ea1bc6961e3c`, both assigned to `train`;
  `tests/unit/test_splits.py::test_identical_texts_always_share_a_split` (50 seeds);
  `tests/data/test_prepared_cuad.py::test_identical_texts_share_a_split`. Commit `298c197`.
- **Interview version:** "I hashed every contract and found that two of CUAD's 510 contracts
  are the same document under different titles — and the annotators labeled them differently on
  two clauses. I made the dev split sample by document identity rather than by title, so
  duplicates can't straddle a split, and an identical text crossing train and test fails loudly."

## LOG-004 — The seeded dev split is longer and richer in liability caps than the test split

- **Date / Phase:** 2026-10-20 / Phase 1
- **Category:** surprising result / decision
- **Context:** First run of `scripts/prepare_cuad.py` with the agreed dev split (102 contracts
  from official train, seed 42).
- **The problem:** The random dev draw differs noticeably from the official test split. Median
  contract length is 44,508.5 characters in dev vs 25,657 in test (33,701 in the remaining
  train). `Cap On Liability` is answerable in 64/102 dev contracts vs 44/102 test contracts.
- **How it was found:** Reading `stats.json` per split right after the first preparation run.
- **Impact if missed:** Parameters tuned on dev (chunk size, candidate depth) are chosen on
  longer documents than the ones they will be tested on; latency per contract will look higher
  on dev, and dev and test metrics are not directly comparable. Re-drawing the seed until dev
  "looks like" test would be a worse error: a choice made after looking at the data, which
  undermines the claim that the split is unbiased.
- **Fix / decision:** Keep seed 42 as agreed before seeing the data. Document the shift in
  `docs/DATASET.md` and report dev and test results separately, with these differences stated
  next to them. The official test split is itself shorter than train, so some shift is inherent
  to CUAD, not only to our draw.
- **Evidence:** `data/processed/cuad/stats.json` (`contract_chars_per_split`,
  `labels.Cap On Liability`), summarized in `docs/DATASET.md`. Commit `4fa3c5f`.
- **Interview version:** "My dev split turned out to have longer contracts than the test set —
  median 44k characters versus 26k. I deliberately did not re-roll the seed, because picking a
  seed after looking at the data is a quiet form of overfitting the evaluation. Instead I
  documented the shift and report dev and test results separately."

## LOG-005 — U+2028 inside a contract split a JSONL record in two

- **Date / Phase:** 2026-10-20 / Phase 1
- **Category:** bug
- **Context:** Writing prepared contracts as JSONL (one JSON object per line) and loading them
  back.
- **The problem:** Loading the real prepared file failed with `Invalid JSON: EOF while parsing
  a string`. The contract `KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT` contains
  16 occurrences of U+2028 LINE SEPARATOR (the only boundary-like character found in any of the
  510 contracts). `json.dumps(..., ensure_ascii=False)` escapes control characters such as
  `\f`, but leaves U+2028 raw, and Python's `str.splitlines()` treats U+2028 as a line break —
  so one record became several invalid "lines".
- **How it was found:** The `data`-marked integrity tests on the real dataset failed, while all
  unit tests on the synthetic fixture passed (the fixture had no such character). A scan of all
  contexts for every character `splitlines()` treats as a boundary located the cause.
- **Impact if missed:** The loader crashed here, which is the good case. A tool that splits
  lines differently (or skips bad lines) would silently drop or corrupt a contract, and a
  hand-written JSONL reader in another language could disagree with ours.
- **Fix / decision:** Two independent safeguards: outputs are written with `ensure_ascii=True`
  (every file is pure ASCII, so no reader can split a record anywhere but at `\n`), and loaders
  split on `"\n"` only, as JSONL defines. The synthetic fixture now includes U+2028 and
  non-ASCII text; without the fix, 5 unit tests fail, with it all pass. Escaping grew
  `contracts.jsonl` by 24,554 bytes (0.09%).
- **Evidence:** commit `d8b2e96`; `tests/unit/test_store.py::test_files_are_pure_ascii` and
  `::test_jsonl_has_one_line_per_record`; preparation re-verified byte-identical across runs
  after the change.
- **Interview version:** "One CUAD contract contains the Unicode LINE SEPARATOR character, which
  Python's JSON encoder doesn't escape but `splitlines()` treats as a newline, so a single
  record was split in two. My synthetic tests missed it; my real-data tests caught it. I made
  every output file pure ASCII and added that character to the fixture, so the regression is now
  covered without needing the full dataset."

## LOG-006 — BM25 implemented in-house with Lucene's idf instead of using `rank-bm25`

- **Date / Phase:** 2026-10-20 / Phase 2
- **Category:** decision
- **Context:** Choosing the BM25 implementation for the baseline. The brief lists `rank-bm25` as
  acceptable. Our indexes are unusual: one index per contract (retrieval never leaves the
  contract), with a median of 52 chunks per train contract (min 2, max 456).
- **The problem:** This is a deliberate design choice, not a library bug. `rank-bm25`'s
  `BM25Okapi` uses the classic Robertson–Spärck Jones idf, `ln(N − n + 0.5) − ln(n + 0.5)`,
  which is negative for any term present in more than half of the documents; it then replaces
  those values with `epsilon × average idf of the whole vocabulary` (read in the library source,
  version 0.2.2, the latest release, from 2022, which also requires numpy). That is a reasonable
  convention for large corpora, but in a small per-contract index it makes some term weights
  depend on the contract's entire vocabulary, which is harder to reason about and to test.
- **How it was found:** Reading the `BM25Okapi._calc_idf` source, then measuring on the 306
  train contracts (no dev or test data) with our query templates and 150/50 chunking.
- **Impact if missed:** Measured: in all 306 train contracts at least one query term would get
  a negative classic idf (median 10 of 70 query terms per contract). Honest nuance: the affected
  terms are overwhelmingly function words (`the` 305 contracts, `of` 304, `to` 303, `shall` 258,
  `not` 173); content terms are rarely affected (`terms` 12, `written` 5, `term` 5, most others
  1–2). So the practical ranking difference is probably small; the cost would have been
  opacity, not broken retrieval.
- **Fix / decision:** A ~60-line BM25 using Lucene's idf, `ln(1 + (N − n + 0.5) / (n + 0.5))`,
  which is always positive and depends only on N and n; Lucene's defaults `k1 = 1.2`,
  `b = 0.75`; distinct query terms counted once; chunks with zero score never returned. Scores
  are tested against values written out from the formula by hand. No numpy dependency.
- **Evidence:** `src/retrieval_labs/retrieval/bm25.py`; `tests/unit/test_bm25.py`
  (`test_scores_match_hand_computed_values`, `test_idf_stays_positive_for_terms_in_every_chunk`);
  measurement script output recorded above.
- **Interview version:** "BM25 is simple enough to own, and our setting is unusual: each index is
  a single contract of about fifty chunks. The popular library uses an idf that goes negative for
  terms in more than half the documents and patches it with a vocabulary-wide average. I measured
  that this mostly touches stopwords, so the gain isn't accuracy; it's that every score in my
  system is a formula I can write on a whiteboard and test by hand."

## LOG-007 — `messages.parse` would hide the cost of truncated or refused calls

- **Date / Phase:** 2026-10-20 / Phase 2
- **Category:** risk / decision
- **Context:** Implementing the Claude extractor with structured outputs. The SDK's recommended
  helper is `client.messages.parse(output_format=PydanticModel)`. Every paid call must be
  recorded in the spend ledger, because the project has a hard US$ 20 budget.
- **The problem:** In `anthropic` 1.15.0, `parse` validates the JSON of every text block inside
  the call (`TypeAdapter.validate_json` in `lib/_parse/_response.py`). A response cut off by
  `max_tokens`, or any text that does not match the schema, raises `ValidationError` from the
  call itself — so the caller never receives the message, its `stop_reason`, or its `usage`,
  even though the call was billed.
- **How it was found:** Reading the installed SDK source before choosing between `parse` and
  `create`, then demonstrating it offline: a `Message` with `stop_reason="max_tokens"`, a
  truncated JSON text block and `usage.output_tokens=4000` passed to the SDK's
  `parse_response` raised `pydantic_core.ValidationError`.
- **Impact if missed:** Truncated calls are the most expensive ones (they used the whole
  `max_tokens`), and they would be exactly the calls missing from the ledger, so the budget
  guard would under-count spend precisely when it matters. The pipeline would also crash
  instead of reporting the clause as `UNCERTAIN`.
- **Fix / decision:** Call `messages.create` with
  `output_config={"effort": ..., "format": {"type": "json_schema", "schema":
  anthropic.transform_schema(...)}}` (the same schema conversion `parse` uses), read `usage`
  first, check `stop_reason` (`refusal`, `max_tokens`), and only then validate the JSON. Every
  failure returns a response with usage and a reason; the clause becomes `UNCERTAIN`.
- **Evidence:** `src/retrieval_labs/extraction/anthropic_extractor.py`;
  `tests/unit/test_anthropic_extractor.py::test_failures_keep_usage_so_cost_is_never_lost`
  (truncated, refused, schema-invalid and empty responses all keep their cost).
- **Interview version:** "The SDK's parse helper validates the JSON inside the call, so a
  truncated response throws before you can read its token usage. Those are the most expensive
  calls, and they'd be invisible to my budget guard. I call the lower-level endpoint with the
  same schema, record usage first, and validate afterwards."

## LOG-008 — BM25 scores depended on Python's hash seed

- **Date / Phase:** 2026-10-20 / Phase 3
- **Category:** bug
- **Context:** Verifying that the retrieval benchmark is byte-reproducible by running it twice
  and comparing every output file.
- **The problem:** `summary.json` was identical across runs, but the per-item results for the
  50/17 configuration were not. BM25 summed per-term contributions iterating over a `set` of
  query terms. The iteration order of a set of strings depends on `PYTHONHASHSEED`, which Python
  randomizes per process, and floating-point addition is not associative, so the same chunk got
  slightly different scores in different processes. Near-tied chunks could swap ranks.
- **How it was found:** The byte-comparison of two benchmark runs flagged one items file. A
  fingerprint of all dev scores and rankings under `PYTHONHASHSEED` = 1, 2, 3 then gave three
  different score fingerprints and two different ranking fingerprints.
- **Impact if missed:** Benchmark numbers that change from run to run with no code change; a
  rank swap near a cutoff changes Recall@k or MRR, and a reviewer re-running the benchmark could
  not reproduce the published figures. The summary was unaffected this time only by luck.
- **Fix / decision:** Sum terms in sorted order. Regression test runs BM25 in subprocesses with
  six hash seeds and requires identical output; it failed before the fix and passes after. The
  benchmark was re-run and verified byte-identical under two different hash seeds.
- **Evidence:** commit `3296c55`; `tests/unit/test_bm25_determinism.py`; real-data fingerprints
  identical for seeds 1–3 after the fix.
- **Interview version:** "My benchmark summary matched between runs, but a byte-level comparison
  of the per-item files didn't. The cause was summing BM25 term scores over a Python set: set
  order changes with the hash seed, and float addition isn't associative, so scores drifted in
  the last digits and near-ties swapped. Sorting the terms fixed it, and a subprocess test with
  several hash seeds keeps it fixed."

## LOG-009 — CUAD's "Cap On Liability" annotations are broader than its description

- **Date / Phase:** 2026-10-20 / Phase 3
- **Category:** surprising result / data quality
- **Context:** Manual failure analysis of the first BM25 benchmark on dev (150/50 chunks):
  all 15 items missed at Recall@5.
- **The problem:** 14 of 15 misses are `Cap On Liability`. In most of them the gold span is not
  a cap at all in the description's sense ("maximum amount for recovery or time limitation to
  bring claims"): 8 are exclusions of indirect/consequential damage types, 2 are "sole remedy"
  clauses, 2 are liability-allocation clauses, 1 a liquidated-damages statement. Our query and
  our extraction prompt were both written from the official description.
- **How it was found:** Reading every miss with its gold text and top-3 retrieved chunks
  (`data/analysis/phase3-bm25-dev/`, dev split only).
- **Impact if missed:** `Cap On Liability` looks like a weak spot of BM25 when it is largely a
  definition mismatch. The same mismatch will affect extraction: a model following the prompt's
  definition may correctly answer NOT_FOUND for a damages exclusion that CUAD counts as a cap,
  which the metrics would score as an error.
- **Fix / decision:** No change in Phase 3 (queries frozen). Documented in `docs/EVALUATION.md`.
  For Phase 4: decide, on dev and documented, whether the query and the extraction definition
  should follow CUAD's operational scope; apply any query change identically to all methods.
- **Evidence:** `reports/benchmark/phase3-bm25-dev/report.md` (Cap On Liability R@5 0.781,
  MRR 0.664 vs ≥ 0.970 R@5 for every other label at 150/50); failure table in
  `docs/EVALUATION.md`.
- **Interview version:** "Almost all my retrieval misses were one label, so I read every one.
  The annotators had marked consequential-damages exclusions and sole-remedy clauses as 'cap on
  liability', while my query followed the dataset's written definition. It was a labeling-scope
  mismatch, not a ranking problem — the kind of thing you only find by reading failures, not
  by staring at the average."

## LOG-010 — Queries kept frozen for Phase 4; query expansion reported only as a separate row

- **Date / Phase:** 2026-10-20 / end of Phase 3 (owner decision)
- **Category:** decision
- **Context:** LOG-009 showed that BM25's `Cap On Liability` misses come from a vocabulary gap:
  CUAD annotates damage exclusions and sole-remedy clauses as caps, while our query (and the
  extraction prompt) follow CUAD's written description. Expanding the query with those terms
  would very likely fix many BM25 misses on dev.
- **The problem:** Phase 4's research question is whether dense and hybrid retrieval close
  gaps that lexical matching cannot. Tuning the shared query on dev now would remove exactly
  the gap that question is about, and it would be tuning on the split we report.
- **How it was found:** Owner review of the Phase 3 failure analysis.
- **Impact if missed:** The four-method comparison would understate the value of semantic
  retrieval, and BM25's dev numbers would be optimistic because its query was fitted to the
  dev failures.
- **Fix / decision:** Queries and the extraction prompt stay frozen, aligned with CUAD's written
  description, for the main comparison of all four methods. After that comparison, an optional
  extra row "BM25 + expanded query" may be added, explicitly marked as tuned on dev and reported
  separately from the fair comparison.
- **Evidence:** `retrieval/queries.py` and `extraction/prompt.py` unchanged since Phase 2
  (`PROMPT_VERSION` and `queries_sha256` recorded in every run); this entry.
- **Interview version:** "I found that most BM25 misses were a vocabulary gap, and fixing the
  query would have been easy. I deliberately didn't: the whole point of comparing dense and
  hybrid retrieval was to see whether they close that gap on their own. The tuned query is
  reported as a separate, clearly labeled row, so nobody confuses a fitted baseline with a fair
  one."

## LOG-011 — The "two-week-old releases" convention was not applied before Phase 4

- **Date / Phase:** 2026-10-20 / Phase 4
- **Category:** risk / decision
- **Context:** Adding the Phase 4 dependencies (sentence-transformers, CPU torch) with uv's
  `exclude-newer` set to 2026-10-06, so only releases at least two weeks old resolve. This is a
  supply-chain caution convention introduced during the project; it is not in the brief (the
  Phase 4 plan wrongly called it a "project rule").
- **The problem:** The global cutoff made the existing lock unsatisfiable. Checking every locked
  package against PyPI upload dates showed 9 of the Phase 0–2 dependencies were published after
  the cutoff, e.g. `anthropic` 1.15.0 (2026-10-15, installed 5 days after release),
  `pydantic` 2.14.1 (2026-10-11), `ruff` 0.17.1 (2026-10-15). The convention had simply never
  been applied before.
- **How it was found:** `uv add` failed resolution under the cutoff; a script then listed the
  upload date of every locked version instead of fixing conflicts one at a time.
- **Impact if missed:** A rule claimed but not enforced. Applying it retroactively would have
  downgraded a tested environment (pydantic, ruff, the Anthropic SDK whose behavior LOG-007
  verified) just to satisfy a convention.
- **Fix / decision (owner):** The convention applies to packages added from Phase 4 on; the
  Phase 0–2 lock stays unchanged. New direct dependencies are pinned (sentence-transformers
  6.1.0, torch 2.14.1, numpy 2.5.3), and every new transitive dependency was checked: 6 were
  newer than the cutoff (certifi, cuda-bindings, filelock, narwhals, setuptools, tokenizers) and
  are pinned via `[tool.uv] constraint-dependencies` to their latest pre-cutoff release, along
  with transformers 5.18.0. Result: 54 new locked packages, none newer than the cutoff, and no
  pre-existing pin changed.
- **Evidence:** `pyproject.toml` `[tool.uv]`; date check of all 54 new lock entries; full test
  suite (177 passed) and the extraction dry run (US$ 0.0937) unchanged after the install.
- **Interview version:** "I had a rule to only install releases at least two weeks old, but when
  I enforced it with the package manager, it showed I'd never actually applied it. Rather than
  silently downgrade a tested environment, I made the rule apply from that point on, pinned the
  six recent transitive dependencies explicitly, and verified every new package's release date."

## LOG-012 — RRF fusion scores are useless as a presence signal

- **Date / Phase:** 2026-10-20 / Phase 4
- **Category:** surprising result
- **Context:** Phase 4 dev benchmark (`reports/benchmark/phase4-dev/`). Presence detection uses
  the top-1 retrieval score against a threshold fit on train, the same rule for every method.
- **The problem:** Hybrid retrieval matches BM25 on ranking (R@5 0.937 for both) but its presence
  detection collapses: macro AUROC 0.597, against 0.842 for BM25, 0.796 for dense and 0.883 for
  hybrid + reranking. Reciprocal rank fusion scores `sum 1 / (k + rank)` depend only on ranks, so
  the top-1 score says where the chunk ranked in each list, not how well it matched.
- **How it was found:** The presence table in the run report. Counting distinct
  `top1_score` values in `items-hybrid.jsonl`: 510 dev items take only 31 distinct values, and
  172 of them tie at the maximum 2/61 = 0.032787 (rank 1 in both component lists).
- **Impact if missed:** Choosing hybrid for the application because it ranks as well as BM25
  would silently degrade the "clause absent" decision, which drives `NOT_FOUND`.
- **Fix / decision:** No change during the dev comparison (methods are frozen before the test
  run). The result is reported as is. Reranking restores a content-based score, which is one
  reason hybrid + reranking has the best presence AUROC. For a hybrid-only configuration,
  presence would need a component score (e.g. BM25 or dense top-1), to be decided later, not
  tuned on dev.
- **Evidence:** `reports/benchmark/phase4-dev/report.md` (presence tables);
  `src/retrieval_labs/retrieval/hybrid.py` (fusion formula); two runs with different
  `PYTHONHASHSEED` gave byte-identical `summary.json` and item files.
- **Interview version:** Hybrid search with reciprocal rank fusion ranked as well as BM25, but
  its scores were almost constant because RRF only looks at ranks, so they could not tell
  whether a clause existed. A good ranking metric is not a good confidence signal; the
  cross-encoder reranker fixed it by re-scoring content.

## LOG-013 — Single test run: dev-tuned query gain halves, reranking becomes significant

- **Date / Phase:** 2026-10-20 / Phase 4
- **Category:** surprising result / decision
- **Context:** The single test-split run (`reports/benchmark/phase4-test/`, commit `022de41`,
  clean tree), five methods frozen after the dev comparison, authorized by the owner.
- **The problem:** Two things had to be checked on unseen contracts: whether the query expanded
  from the dev failure analysis (LOG-010) only memorized dev, and whether the small,
  non-significant dev gain of hybrid + reranking was real.
- **How it was found:** Paired contract-level bootstrap against BM25 on test, compared with
  the same intervals on dev.
- **Impact if missed:** Reporting the dev numbers of `bm25-expanded` would overstate its gain
  about twofold; dropping reranking on dev evidence would have discarded the best fair method.
- **Fix / decision:** Both kept as reported. `bm25-expanded` gains roughly halved but stayed
  positive: ΔR@1 +0.084 [+0.051, +0.119] on dev vs +0.045 [+0.020, +0.077] on test; ΔR@5
  +0.054 vs +0.030 [+0.009, +0.058]. Hybrid + reranking went from ΔR@5 +0.025 [+0.000, +0.052]
  on dev to +0.035 [+0.010, +0.067] on test, and ΔMRR@10 +0.037 [+0.005, +0.070]. The test split
  is now spent; no parameter may be changed using these numbers.
- **Evidence:** `docs/EVALUATION.md` (Phase 4 results); repeating the test run with
  `PYTHONHASHSEED=97` gave byte-identical `summary.json`, item files and failure samples.
  Warm-cache query medians: BM25 0.22 ms, `bm25-expanded` 0.31 ms, hybrid + reranking 431 ms.
- **Interview version:** I tuned one query on dev and kept it out of the fair comparison;
  on the held-out test its gain was about half the dev gain, which is the optimism you expect
  from tuning on the evaluation set. The cross-encoder reranker, which was not significant on
  dev, was significant on test. A cheap, targeted query fix matched the neural reranker at
  roughly 1/1400 of the query latency.

## LOG-014 — The demo's first latency figure included model loading

- **Date / Phase:** 2026-10-20 / minimum publishable version
- **Category:** bug
- **Context:** The Streamlit demo reports how long retrieval took for the selected contract.
- **The problem:** The timer started before the cached `load_retriever` call, so the first
  view of the first contract counted loading both models as retrieval time.
- **How it was found:** A Streamlit `AppTest` run printed "retrieval with `hybrid+rerank`
  took 20.3 s" for a 13,038-character contract whose embeddings were already cached.
- **Impact if missed:** The demo would have shown a latency ten times the measured one,
  contradicting the README and `docs/EVALUATION.md`.
- **Fix / decision:** Load the contract and the retriever before starting the timer. The same
  contract then showed 3.25 s; two others 2.15 s (39,727 characters) and 0.62 s (3,595
  characters). Caption now states what the time includes. Fixed before the first commit of the
  demo.
- **Evidence:** `demo/streamlit_app.py` (`retrieve`); AppTest output before and after.
- **Interview version:** My demo showed 20 seconds for a step I had measured at about two,
  because the timer also covered lazy model loading. Every number a UI shows is a
  measurement claim, so it needs the same scrutiny as the benchmark.

## LOG-015 — Streamlit listens on the network and sends usage statistics by default

- **Date / Phase:** 2026-10-20 / minimum publishable version
- **Category:** risk
- **Context:** First headless start of the demo with `streamlit run`.
- **The problem:** The startup log printed a Network URL (`0.0.0.0` and `::` listeners, per
  `netstat`) and an External URL, and "Collecting usage statistics". A demo that shows contract
  text was reachable from the local network, and the framework phoned home, neither of which
  the brief allows without a decision (Section 18; deployment is Phase 7).
- **How it was found:** The Streamlit startup log and `netstat -ano` for the port.
- **Impact if missed:** Anyone on the same network could open the demo; telemetry left the
  machine without the owner knowing.
- **Fix / decision:** `.streamlit/config.toml` sets `server.address = "localhost"` and
  `browser.gatherUsageStats = false`. After the change, the only listener was `127.0.0.1`, the
  log showed a single local URL, and the health endpoint answered `ok` via `localhost`.
- **Evidence:** `.streamlit/config.toml`; `Get-NetTCPConnection` output before and after.
- **Interview version:** Running the demo for the first time, I noticed Streamlit bound to all
  interfaces and enabled telemetry by default. For a tool that displays contracts, I made it
  local-only and opt-out of telemetry in versioned config, and checked the listeners after.

## LOG-016 — On Linux, the locked torch pulled about 2.4 GB of CUDA packages

- **Date / Phase:** 2026-10-21 / CI (before Phase 6)
- **Category:** performance / decision
- **Context:** Planning the GitHub Actions workflow, which runs on Linux.
- **The problem:** The torch 2.14.1 entry in `uv.lock` comes from PyPI, whose Linux wheel
  depends on 19 packages marked `sys_platform == 'linux'` (`nvidia-*`, `cuda-*`, `triton`).
  Their x86_64 wheels in the lock add up to about 2.45 GB, on top of the 555 MB torch wheel.
  On Windows the PyPI wheel is already CPU-only (124 MB), so local use never showed it.
- **How it was found:** Reading torch's dependencies in `uv.lock` and summing the recorded
  wheel sizes; checking that `download.pytorch.org/whl/cpu` serves
  `torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` (196 MB).
- **Impact if missed:** Every CI run, and any Linux install or deploy, would download and
  cache gigabytes of GPU libraries the project never uses (all models run on CPU, and the
  tests never import torch).
- **Fix / decision:** An explicit `pytorch-cpu` index in `pyproject.toml`, used for torch
  only when `sys_platform == 'linux'`. After `uv lock`, the 19 packages were removed, Linux
  resolves `torch==2.14.1+cpu`, the Windows wheel URL and hash are unchanged, and no other
  package version changed. `uv sync --locked` and the full test suite (226 passed) still pass
  on Windows.
- **Evidence:** `uv lock` output; `pyproject.toml` (`[tool.uv.sources]`); `uv.lock` diff.
- **Interview version:** Setting up CI, I noticed the lockfile would make Linux install CUDA,
  about 2.4 GB, for a CPU-only project, because the Windows wheel I used locally hides it.
  A platform-scoped source sends Linux to PyTorch's CPU index while the Windows lock entry
  stays byte-identical.

## LOG-017 — With Redis down, a contract upload hung for 49 seconds

- **Date / Phase:** 2026-10-21 / Phase 6, step 4
- **Category:** performance / bug
- **Context:** API test for the case where the job queue is unreachable: the upload must be
  stored, its job marked `FAILED` (`queue_unavailable`), and the client answered `503`.
- **The problem:** The test passed but took 48.8 s. redis-py 8 clients retry failed
  connections 10 times with exponential backoff by default, so `POST /contracts` held the
  request for that long before answering.
- **How it was found:** `pytest --durations` flagged the test; timing `Redis(port=1).ping()`
  on its own gave 49.2 s by default, 25.5 s with a 1 s connect timeout, and 2.0 s with a 1 s
  connect timeout and no retries.
- **Impact if missed:** Any Redis outage would make every upload hang for close to a minute
  instead of failing clearly, tying up server threads.
- **Fix / decision:** The API builds its Redis client with `api_redis` (2 s connect timeout,
  5 s socket timeout, one retry, no backoff). The test now takes 4.0 s and asserts it stays
  under 10 s. The worker keeps redis-py's default, where waiting out a Redis restart is right.
- **Evidence:** `src/retrieval_labs/api/app.py` (`api_redis`);
  `tests/integration/test_api.py::test_unavailable_queue_fails_the_job_and_returns_503`.
- **Interview version:** A test of the "queue is down" path passed, but slowly: 49 seconds.
  The Redis client's default retry policy, sensible for a background worker, was wrong for
  a request handler. I gave the API its own fail-fast client and pinned the time in the test.
