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
