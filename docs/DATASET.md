# Dataset — CUAD v1

This document records what was **verified by inspecting the actual files**, not what was assumed
from documentation. Raw-file numbers come from `scripts/inspect_cuad.py` (Phase 0); numbers for
the prepared dataset come from `scripts/prepare_cuad.py` (`stats.json`, `manifest.json`, Phase 1).

## Source and license

- **Dataset:** Contract Understanding Atticus Dataset (CUAD), version 1, by The Atticus Project.
- **Homepage:** https://www.atticusprojectai.org/cuad
- **Repository:** https://github.com/The-Atticus-Project/cuad
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0). Use is also subject to The Atticus
  Project's privacy policy and disclaimer (linked from the homepage).
- **Citation:**

  > Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. *CUAD: An Expert-Annotated NLP Dataset
  > for Legal Contract Review.* arXiv:2103.06268, 2021.

The underlying contracts are public filings (SEC EDGAR exhibits). This repository **does not
redistribute** the dataset: it is downloaded locally into `data/raw/` (git-ignored).

## Selected source file

We use the official `data.zip` from the GitHub repository, because it is the only distribution
that ships the **official contract-level train/test split** (the same archive is used by the
Hugging Face `cuad-qa` loader).

| Property | Value |
|---|---|
| URL | `https://github.com/The-Atticus-Project/cuad/raw/main/data.zip` |
| Size | 18,309,308 bytes |
| SHA-256 | `f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a` |
| Local path | `data/raw/cuad/data.zip` |

Archive contents:

| File | Bytes | Contracts | qa entries | Notes |
|---|---:|---:|---:|---|
| `train_separate_questions.json` | 35,065,844 | 408 | 22,450 | Official train split; multi-answer questions split into separate entries |
| `test.json` | 7,378,232 | 102 | 4,182 | Official test split; 41 questions per contract |
| `CUADv1.json` | 40,128,638 | 510 | 20,910 | All contracts; 41 questions per contract |

Other distributions (not used for the benchmark): the Hugging Face repository
`theatticusproject/cuad` holds `CUAD_v1/CUAD_v1.json`, `master_clauses.csv/.xlsx`, the per-group
Excel files, and the raw `full_contract_txt/` and `full_contract_pdf/` folders.

### Download and preparation

```bash
uv run python scripts/prepare_cuad.py --download   # downloads (if absent), verifies, prepares
uv run python scripts/inspect_cuad.py data/raw/cuad/data.zip   # optional independent cross-check
```

`--download` fetches the archive only if it is absent; an existing file with the wrong SHA-256
is reported and never overwritten. See "Prepared dataset" below for the outputs.

## Schema (SQuAD 2.0 style)

```text
{
  "version": "aok_v1.0",
  "data": [                                  # one entry per contract
    {
      "title": "<contract name>",            # unique per file; used as contract ID
      "paragraphs": [                        # always exactly one paragraph
        {
          "context": "<full contract text>",
          "qas": [
            {
              "id": "<title>__<Label>",      # train file: "<title>__<Label>_<n>"
              "question": "Highlight the parts (if any) of this contract related to \"<Label>\" that should be reviewed by a lawyer. Details: <description>",
              "is_impossible": true|false,   # true iff answers is empty (verified)
              "answers": [{"text": "...", "answer_start": <int>}]
            }
          ]
        }
      ]
    }
  ]
}
```

Verified properties:

- Every contract has exactly one `paragraph`; its `context` is the whole contract text.
- The label in the `id` always equals the quoted label in the `question`.
- Every contract has a question for **every** label; an absent clause is represented by
  `answers: []` and `is_impossible: true` (no inconsistencies found).
- In `train_separate_questions.json`, a label can appear in up to 55 entries for one contract
  (`<Label>_0`, `<Label>_1`, ...). After merging entries by label, the answer sets are
  **identical** to `CUADv1.json` for every contract and label.
- Contexts are identical between the split files and `CUADv1.json`.

## Split integrity

| Check | Result |
|---|---|
| Train contracts | 408 |
| Test contracts | 102 |
| Contracts in both train and test | 0 |
| Train ∪ test equals the 510 contracts of `CUADv1.json` | yes |

The dev split derived from the 408 training contracts is described under "Prepared dataset"
below. The 102 test contracts are reserved for final evaluation only.

## Offset integrity

For every answer span, `context[answer_start : answer_start + len(text)] == text`:

| File | Answer spans | Exact match | Mismatch |
|---|---:|---:|---:|
| train | 11,180 | 11,180 | 0 |
| test | 2,643 | 2,643 | 0 |
| full | 13,823 | 13,823 | 0 |

Also verified: no context and no answer text has leading/trailing whitespace, and no context
contains `\r\n`. The Hugging Face `cuad-qa` loader calls `.strip()` on contexts and answers
without adjusting offsets; this happens to be harmless on the current data, but we still read
the raw JSON so that offsets are guaranteed to refer to the original text.

## Contract length

| Split | Chars median | Chars p95 | Chars max | Whitespace tokens median | Tokens max |
|---|---:|---:|---:|---:|---:|
| train | 35,960 | 163,199 | 338,211 | 5,358 | 45,650 |
| test | 25,657 | 145,168 | 300,768 | 3,911 | 47,733 |

Long contracts make full-context LLM calls expensive and motivate retrieval over chunks.

## Target labels

The project's label names **match the dataset exactly** (Title Case, as in `id` and `question`).
Note that `category_descriptions.csv` in the GitHub repository uses different capitalization
(e.g. "Cap on Liability"); the JSON spelling is authoritative for this project.

| Product category | CUAD label (exact) |
|---|---|
| Governing Law | `Governing Law` |
| Termination | `Termination For Convenience` |
| Liability | `Cap On Liability` |
| Renewal | `Renewal Term` |
| Renewal notice | `Notice Period To Terminate Renewal` |

### Coverage per split

| Label | Split | With answer | Without answer | Multi-span contracts | Total spans | Max spans | Median span chars |
|---|---|---:|---:|---:|---:|---:|---:|
| Governing Law | train | 354 | 54 | 20 | 374 | 2 | 180 |
| Governing Law | test | 83 | 19 | 5 | 90 | 4 | 177 |
| Termination For Convenience | train | 154 | 254 | 41 | 205 | 4 | 160 |
| Termination For Convenience | test | 29 | 73 | 9 | 41 | 4 | 177 |
| Cap On Liability | train | 231 | 177 | 133 | 554 | 16 | 313 |
| Cap On Liability | test | 44 | 58 | 28 | 118 | 9 | 284 |
| Renewal Term | train | 150 | 258 | 22 | 179 | 5 | 273 |
| Renewal Term | test | 26 | 76 | 3 | 31 | 4 | 234 |
| Notice Period To Terminate Renewal | train | 95 | 313 | 8 | 104 | 3 | 298 |
| Notice Period To Terminate Renewal | test | 16 | 86 | 2 | 18 | 2 | 294 |

### Implications for later phases

- **Small test positives.** Some labels have few answerable test contracts (16 for renewal
  notice, 26 for renewal term, 29 for termination for convenience). Per-label metrics on the test
  set will have wide uncertainty; this must be reported, not hidden.
- **Absence is common.** Most labels are absent in most contracts, so presence/absence detection
  (Section 9.3 of the brief) is a substantial part of the evaluation, separate from answerable-case
  retrieval metrics.
- **Multiple spans.** `Cap On Liability` frequently has several gold spans (up to 16 in one
  contract), which the relevance criterion and extraction metrics must handle explicitly.
- **Span length.** Median gold spans are roughly 160–310 characters, which is relevant when
  choosing chunk sizes (Addendum 21.6).

## Prepared dataset (Phase 1)

Produced by `scripts/prepare_cuad.py` into `data/processed/cuad/` (git-ignored). Text and
annotations come from `CUADv1.json`; the split files are used only for membership.

| File | Records | Content |
|---|---:|---|
| `contracts.jsonl` | 510 | `contract_id`, `title`, `text`, `text_sha256` — **no annotation fields** |
| `gold_annotations.jsonl` | 2,550 | One per (contract, target label), including empty ones: `contract_id`, `label`, `spans[{start, end, text}]` (`end` exclusive) |
| `stats.json` | 1 | Per-split, per-label statistics |
| `manifest.json` | 1 | Source URL, archive SHA-256, JSON version, generator version, seed, dev size, split of every contract, duplicate-text groups, SHA-256 and record count of every output file |

Properties (all covered by tests):

- **Reproducible:** two runs produce byte-identical files (no timestamps; sorted records and keys).
- **Pure ASCII, LF:** non-ASCII characters are JSON-escaped, so records can only break at `\n`
  (one contract contains U+2028 LINE SEPARATOR; see ENGINEERING_LOG LOG-005).
- **Hash-verified loading:** `load_contracts` / `load_gold` refuse files whose SHA-256 differs
  from the manifest.
- **Test lock:** `load_gold(Split.TEST)` raises unless `final_evaluation=True`.
- `contract_id` is `cuad_` + the first 12 hex digits of SHA-256(title).

### Splits

| Split | Contracts | Source |
|---|---:|---|
| train | 306 | Official train minus dev — exploration, examples |
| dev | 102 | Sampled from official train, seed 42 — all tuning |
| test | 102 | Official test — final evaluation only |

The dev split is sampled by **document identity**: contracts with byte-identical text form one
group and always share a split. CUAD has one such pair, both in official train:
`cuad_8c7e7f51de1f` and `cuad_ea1bc6961e3c` (`ADUROBIOTECH…CONSULTING AGREEMENT` with and
without a `(1)` suffix). Their annotations differ on two non-target labels (LOG-003).
Near-duplicates are not detected.

### Statistics per split

Contract length in characters:

| Split | Median | Mean | p95 | Max |
|---|---:|---:|---:|---:|
| train | 33,701 | 51,461.2 | 161,066 | 338,211 |
| dev | 44,508.5 | 61,583.2 | 225,094 | 335,282 |
| test | 25,657 | 46,848.2 | 145,168 | 300,768 |

Target labels (A = contracts with an answer, of the split's contracts; Spans = total gold
spans; Multi = contracts with more than one span; Overlap = pairs of spans sharing characters):

| Label | train A / Spans / Multi / Overlap | dev A / Spans / Multi / Overlap | test A / Spans / Multi / Overlap |
|---|---|---|---|
| Governing Law | 265 / 282 / 17 / 0 | 89 / 92 / 3 / 0 | 83 / 90 / 5 / 0 |
| Termination For Convenience | 121 / 157 / 30 / 1 | 33 / 48 / 11 / 0 | 29 / 41 / 9 / 0 |
| Cap On Liability | 167 / 407 / 95 / 7 | 64 / 147 / 38 / 2 | 44 / 118 / 28 / 1 |
| Renewal Term | 118 / 142 / 18 / 2 | 32 / 37 / 4 / 0 | 26 / 31 / 3 / 0 |
| Notice Period To Terminate Renewal | 74 / 82 / 7 / 0 | 21 / 22 / 1 / 0 | 16 / 18 / 2 / 0 |

Totals across all splits: 1,182 answerable (contract, label) pairs and 1,714 spans — identical
to the independent Phase 0 inspection. No exact-duplicate spans exist; 13 span pairs overlap
(5 of them nested), so metrics must not assume disjoint gold spans.

**Dev/test shift (LOG-004):** dev contracts are longer than test contracts (median 44.5k vs
25.7k characters) and `Cap On Liability` is more frequent in dev (64 vs 44 of 102). The seed was
fixed before the data was seen and was not changed. Dev and test results must be reported
separately, with this difference stated.

## All 41 labels (exact spelling in the JSON)

Affiliate License-Licensee, Affiliate License-Licensor, Agreement Date, Anti-Assignment,
Audit Rights, Cap On Liability, Change Of Control, Competitive Restriction Exception,
Covenant Not To Sue, Document Name, Effective Date, Exclusivity, Expiration Date, Governing Law,
Insurance, Ip Ownership Assignment, Irrevocable Or Perpetual License, Joint Ip Ownership,
License Grant, Liquidated Damages, Minimum Commitment, Most Favored Nation,
No-Solicit Of Customers, No-Solicit Of Employees, Non-Compete, Non-Disparagement,
Non-Transferable License, Notice Period To Terminate Renewal, Parties, Post-Termination Services,
Price Restrictions, Renewal Term, Revenue/Profit Sharing, Rofr/Rofo/Rofn, Source Code Escrow,
Termination For Convenience, Third Party Beneficiary, Uncapped Liability,
Unlimited/All-You-Can-Eat-License, Volume Restriction, Warranty Duration.
