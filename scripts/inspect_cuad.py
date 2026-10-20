"""Inspect the official CUAD v1 ``data.zip`` and print a reproducible summary.

Phase 0 exploration tool. It reads the archive directly with the standard
library (no Hugging Face loader, which strips contexts without adjusting
offsets) and reports the facts recorded in ``docs/DATASET.md``:

- archive contents and SHA-256,
- JSON schema and split sizes,
- contract-level overlap between the official train and test splits,
- consistency between the split files and the combined ``CUADv1.json``,
- exact names and coverage of the project's target labels,
- integrity of the answer character offsets.

Usage:
    uv run python scripts/inspect_cuad.py data/raw/cuad/data.zip [--json-out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRAIN_FILE = "train_separate_questions.json"
TEST_FILE = "test.json"
FULL_FILE = "CUADv1.json"

TARGET_LABELS = (
    "Governing Law",
    "Termination For Convenience",
    "Cap On Liability",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
)

QUESTION_LABEL_RE = re.compile(r'related to "(?P<label>[^"]+)"')
SPLIT_SUFFIX_RE = re.compile(r"_(\d+)$")


@dataclass
class Span:
    start: int
    text: str


@dataclass
class Contract:
    title: str
    context: str
    # label -> list of answer spans (merged across "separate question" entries)
    answers: dict[str, list[Span]] = field(default_factory=dict)
    # label -> number of raw qa entries that produced the label
    qa_entries: Counter[str] = field(default_factory=Counter)
    impossible_flags: dict[str, set[bool]] = field(default_factory=lambda: defaultdict(set))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_from_qa(qa: dict[str, Any], title: str) -> str:
    """Derive the label from the qa id and cross-check it against the question text."""
    prefix = f"{title}__"
    if not qa["id"].startswith(prefix):
        raise ValueError(f"qa id does not start with title: {qa['id']!r}")
    id_label = SPLIT_SUFFIX_RE.sub("", qa["id"][len(prefix) :])
    match = QUESTION_LABEL_RE.search(qa["question"])
    if match is None or match.group("label") != id_label:
        raise ValueError(f"label mismatch between id and question: {qa['id']!r}")
    return id_label


def load_contracts(archive: zipfile.ZipFile, name: str) -> dict[str, Contract]:
    payload = json.loads(archive.read(name))
    contracts: dict[str, Contract] = {}
    for doc in payload["data"]:
        title = doc["title"]
        if title in contracts:
            raise ValueError(f"duplicate title in {name}: {title!r}")
        if len(doc["paragraphs"]) != 1:
            raise ValueError(f"expected one paragraph per contract: {title!r}")
        paragraph = doc["paragraphs"][0]
        contract = Contract(title=title, context=paragraph["context"])
        for qa in paragraph["qas"]:
            label = label_from_qa(qa, title)
            contract.qa_entries[label] += 1
            contract.impossible_flags[label].add(bool(qa["is_impossible"]))
            spans = contract.answers.setdefault(label, [])
            spans.extend(Span(a["answer_start"], a["text"]) for a in qa["answers"])
        contracts[title] = contract
    return contracts


def span_set(spans: list[Span]) -> set[tuple[int, str]]:
    return {(s.start, s.text) for s in spans}


def describe(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 1),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def offset_report(contracts: dict[str, Contract]) -> dict[str, int]:
    total = exact = mismatched = edge_whitespace = 0
    for contract in contracts.values():
        for spans in contract.answers.values():
            for span in spans:
                total += 1
                if contract.context[span.start : span.start + len(span.text)] == span.text:
                    exact += 1
                else:
                    mismatched += 1
                if span.text != span.text.strip():
                    edge_whitespace += 1
    return {
        "answer_spans": total,
        "offset_exact_match": exact,
        "offset_mismatch": mismatched,
        "answers_with_edge_whitespace": edge_whitespace,
    }


def label_report(contracts: dict[str, Contract], label: str) -> dict[str, Any]:
    with_answer = [c for c in contracts.values() if c.answers.get(label)]
    spans_per_contract = [len(c.answers[label]) for c in with_answer]
    span_lengths = [len(s.text) for c in with_answer for s in c.answers[label]]
    impossible_inconsistent = sum(
        1
        for c in contracts.values()
        if c.impossible_flags.get(label) and c.impossible_flags[label] != {not c.answers[label]}
    )
    return {
        "contracts": len(contracts),
        "contracts_with_label_entry": sum(1 for c in contracts.values() if label in c.answers),
        "contracts_with_answer": len(with_answer),
        "contracts_without_answer": len(contracts) - len(with_answer),
        "contracts_with_multiple_spans": sum(1 for n in spans_per_contract if n > 1),
        "total_spans": sum(spans_per_contract),
        "max_spans_in_one_contract": max(spans_per_contract, default=0),
        "span_length_chars": describe(span_lengths),
        "is_impossible_inconsistent_with_answers": impossible_inconsistent,
    }


def resolve_label(requested: str, available: set[str]) -> str | None:
    """Return the dataset's exact spelling of a label, matching case-insensitively."""
    matches = [name for name in available if name.casefold() == requested.casefold()]
    return matches[0] if len(matches) == 1 else None


def inspect(zip_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"archive": str(zip_path), "sha256": sha256_of(zip_path)}
    with zipfile.ZipFile(zip_path) as archive:
        report["members"] = {i.filename: i.file_size for i in archive.infolist()}
        train = load_contracts(archive, TRAIN_FILE)
        test = load_contracts(archive, TEST_FILE)
        full = load_contracts(archive, FULL_FILE)

    splits = {"train": train, "test": test, "full": full}
    report["splits"] = {
        name: {
            "contracts": len(contracts),
            "distinct_labels": len({lbl for c in contracts.values() for lbl in c.answers}),
            "qa_entries": sum(sum(c.qa_entries.values()) for c in contracts.values()),
            "context_chars": describe([len(c.context) for c in contracts.values()]),
            "context_whitespace_tokens": describe(
                [len(c.context.split()) for c in contracts.values()]
            ),
            "contexts_with_edge_whitespace": sum(
                1 for c in contracts.values() if c.context != c.context.strip()
            ),
            "contexts_with_crlf": sum(1 for c in contracts.values() if "\r\n" in c.context),
            "offsets": offset_report(contracts),
        }
        for name, contracts in splits.items()
    }

    train_titles, test_titles, full_titles = set(train), set(test), set(full)
    report["split_membership"] = {
        "train_test_overlap": len(train_titles & test_titles),
        "train_plus_test_equals_full": train_titles | test_titles == full_titles,
        "titles_only_in_full": sorted(full_titles - train_titles - test_titles),
    }

    # Is the combined file a faithful merge of the split files?
    context_diffs = 0
    answer_diffs: Counter[str] = Counter()
    for title, contract in full.items():
        source = train.get(title) or test.get(title)
        if source is None:
            continue
        if source.context != contract.context:
            context_diffs += 1
        for label in set(contract.answers) | set(source.answers):
            if span_set(contract.answers.get(label, [])) != span_set(source.answers.get(label, [])):
                answer_diffs[label] += 1
    report["full_vs_splits"] = {
        "context_differences": context_diffs,
        "answer_set_differences_by_label": dict(answer_diffs),
    }

    available = {lbl for c in full.values() for lbl in c.answers}
    report["all_labels"] = sorted(available)
    report["train_split_qa_entries_per_label_max"] = max(
        n for c in train.values() for n in c.qa_entries.values()
    )

    targets: dict[str, Any] = {}
    for requested in TARGET_LABELS:
        exact = resolve_label(requested, available)
        entry: dict[str, Any] = {"dataset_name": exact}
        if exact is not None:
            entry["by_split"] = {
                "train": label_report(train, exact),
                "test": label_report(test, exact),
            }
        targets[requested] = entry
    report["target_labels"] = targets
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip_path", type=Path, help="path to the official CUAD data.zip")
    parser.add_argument("--json-out", type=Path, help="optional path for the JSON report")
    args = parser.parse_args()

    report = inspect(args.zip_path)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
