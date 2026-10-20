"""A tiny synthetic archive in the real CUAD v1 format, for tests without network or dataset."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from retrieval_labs.cuad.labels import CuadLabel

# A non-target label, present in real CUAD, that the parser must ignore.
OTHER_LABEL = "Parties"

ALPHA = "ALPHA_2020-EX-10.1-SUPPLY AGREEMENT"
BETA = "BETA,INC_2019-EX-10.2-SERVICES AGREEMENT/AMENDED(1)"
GAMMA = "GAMMA_2018-EX-10.3-LICENSE"
DELTA = "DELTA_2021-EX-10.4-DISTRIBUTION AGREEMENT"

ALPHA_TEXT = (
    "\n   SUPPLY AGREEMENT\n\n"  # leading whitespace: must be preserved, never stripped
    "1. Term. This Agreement renews automatically for successive one-year periods.\n"
    "2. Notice. Either party may prevent renewal by giving 60 days written notice.\n"
    "3. Liability. Liability is capped at fees paid. "
    "Neither party is liable for indirect damages.\n"
    "4. Law. This Agreement is governed by the laws of the State of New York.\n"
)

# title -> (text, {label: [answer fragments]})
SYNTHETIC: dict[str, tuple[str, dict[str, list[str]]]] = {
    ALPHA: (
        ALPHA_TEXT,
        {
            CuadLabel.GOVERNING_LAW: [
                "This Agreement is governed by the laws of the State of New York."
            ],
            CuadLabel.CAP_ON_LIABILITY: [
                "Neither party is liable for indirect damages.",
                "Liability is capped at fees paid.",  # out of order on purpose
            ],
            CuadLabel.RENEWAL_TERM: [
                "This Agreement renews automatically for successive one-year periods."
            ],
            CuadLabel.NOTICE_PERIOD_TO_TERMINATE_RENEWAL: [
                "Either party may prevent renewal by giving 60 days written notice."
            ],
            OTHER_LABEL: ["SUPPLY AGREEMENT"],
        },
    ),
    BETA: (
        "SERVICES AGREEMENT\nThe parties are Beta Corp and Client LLC.\n",
        {OTHER_LABEL: ["Beta Corp", "Client LLC"]},
    ),
    GAMMA: (
        "LICENSE\nLicensee may terminate this License at any time upon 30 days notice.\n",
        {
            CuadLabel.TERMINATION_FOR_CONVENIENCE: [
                "Licensee may terminate this License at any time upon 30 days notice."
            ]
        },
    ),
    DELTA: (
        "DISTRIBUTION AGREEMENT\nGoverned by the laws of Delaware.\n",
        {CuadLabel.GOVERNING_LAW: ["Governed by the laws of Delaware."]},
    ),
}
SYNTHETIC_TRAIN = [ALPHA, BETA, GAMMA]
SYNTHETIC_TEST = [DELTA]

ArchiveFactory = Callable[..., Path]


def make_qa(text: str, label: str, fragments: list[str], qa_id: str) -> dict[str, Any]:
    return {
        "id": qa_id,
        "question": (
            f'Highlight the parts (if any) of this contract related to "{label}" '
            "that should be reviewed by a lawyer. Details: synthetic"
        ),
        "is_impossible": not fragments,
        "answers": [{"text": f, "answer_start": text.index(f)} for f in fragments],
    }


def make_document(title: str, separate_questions: bool = False) -> dict[str, Any]:
    text, answers = SYNTHETIC[title]
    qas = []
    for label in [*CuadLabel, OTHER_LABEL]:
        fragments = answers.get(label, [])
        if separate_questions and len(fragments) > 1:
            # train_separate_questions.json: one entry per answer, ids suffixed _0, _1, ...
            qas += [
                make_qa(text, label, [f], f"{title}__{label}_{i}") for i, f in enumerate(fragments)
            ]
        else:
            qas.append(make_qa(text, label, fragments, f"{title}__{label}"))
    return {"title": title, "paragraphs": [{"context": text, "qas": qas}]}


def write_archive(
    path: Path,
    mutate: Callable[[list[dict[str, Any]]], None] | None = None,
    train: list[str] = SYNTHETIC_TRAIN,
    test: list[str] = SYNTHETIC_TEST,
) -> Path:
    """Write a synthetic data.zip; ``mutate`` edits the CUADv1.json documents first."""
    full = [make_document(t) for t in SYNTHETIC]
    if mutate is not None:
        mutate(full)
    files = {
        "CUADv1.json": full,
        "train_separate_questions.json": [make_document(t, separate_questions=True) for t in train],
        "test.json": [make_document(t) for t in test],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, docs in files.items():
            archive.writestr(name, json.dumps({"version": "aok_v1.0", "data": docs}))
    return path
