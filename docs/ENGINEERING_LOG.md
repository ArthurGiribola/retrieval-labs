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
