"""Parse the official CUAD v1 archive into source contracts and gold annotations.

Text and annotations come from ``CUADv1.json``; the official split files are read only
for their titles (split membership). Contracts and gold annotations are returned as
separate collections so callers can hand contracts to inference code without labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.cuad.gold import GoldAnnotation, GoldSpan, check_offsets
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.documents import Contract

FULL_FILE = "CUADv1.json"
TRAIN_FILE = "train_separate_questions.json"
TEST_FILE = "test.json"

_QUESTION_LABEL_RE = re.compile(r'related to "(?P<label>[^"]+)"')
_SEPARATE_QUESTION_SUFFIX_RE = re.compile(r"_\d+$")
_LABEL_ORDER = {label: index for index, label in enumerate(CuadLabel)}


@dataclass(frozen=True)
class ParsedCuad:
    json_version: str
    contracts: tuple[Contract, ...]  # sorted by contract_id
    gold: tuple[GoldAnnotation, ...]  # sorted by (contract_id, CuadLabel order)
    official_train_ids: frozenset[str]
    official_test_ids: frozenset[str]


def make_contract_id(title: str) -> str:
    """Stable, filesystem-safe ID derived from the CUAD title."""
    return "cuad_" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]


def parse_archive(path: Path) -> ParsedCuad:
    with zipfile.ZipFile(path) as archive:
        full = _read_json(archive, FULL_FILE)
        train_titles = _titles(_read_json(archive, TRAIN_FILE), TRAIN_FILE)
        test_titles = _titles(_read_json(archive, TEST_FILE), TEST_FILE)

    contracts: list[Contract] = []
    gold: list[GoldAnnotation] = []
    id_by_title: dict[str, str] = {}
    for doc in full["data"]:
        contract, annotations = _parse_document(doc)
        if contract.title in id_by_title:
            raise CuadDataError(f"duplicate title in {FULL_FILE}: {contract.title!r}")
        if contract.contract_id in id_by_title.values():
            raise CuadDataError(f"contract_id collision: {contract.contract_id}")
        id_by_title[contract.title] = contract.contract_id
        contracts.append(contract)
        gold.extend(annotations)

    _check_membership(set(id_by_title), set(train_titles), set(test_titles))
    contracts.sort(key=lambda c: c.contract_id)
    gold.sort(key=lambda g: (g.contract_id, _LABEL_ORDER[g.label]))
    return ParsedCuad(
        json_version=full["version"],
        contracts=tuple(contracts),
        gold=tuple(gold),
        official_train_ids=frozenset(id_by_title[t] for t in train_titles),
        official_test_ids=frozenset(id_by_title[t] for t in test_titles),
    )


def _read_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        return json.loads(archive.read(name).decode("utf-8"))
    except KeyError as exc:
        raise CuadDataError(f"{name} missing from archive") from exc


def _titles(payload: dict[str, Any], name: str) -> list[str]:
    titles = [doc["title"] for doc in payload["data"]]
    if len(titles) != len(set(titles)):
        raise CuadDataError(f"duplicate titles in {name}")
    return titles


def _label_of(qa: dict[str, Any], title: str) -> str:
    """Label from the qa id, cross-checked against the quoted label in the question."""
    prefix = f"{title}__"
    if not qa["id"].startswith(prefix):
        raise CuadDataError(f"qa id does not start with its title: {qa['id']!r}")
    id_label = _SEPARATE_QUESTION_SUFFIX_RE.sub("", qa["id"][len(prefix) :])
    match = _QUESTION_LABEL_RE.search(qa["question"])
    if match is None or match.group("label") != id_label:
        raise CuadDataError(f"label differs between id and question: {qa['id']!r}")
    return id_label


def _parse_document(doc: dict[str, Any]) -> tuple[Contract, list[GoldAnnotation]]:
    title = doc["title"]
    if len(doc["paragraphs"]) != 1:
        raise CuadDataError(f"expected exactly one paragraph: {title!r}")
    paragraph = doc["paragraphs"][0]
    contract = Contract.from_text(make_contract_id(title), title, paragraph["context"])

    spans_by_label: dict[CuadLabel, list[GoldSpan]] = {}
    for qa in paragraph["qas"]:
        name = _label_of(qa, title)
        if name not in CuadLabel:
            continue
        label = CuadLabel(name)
        if label in spans_by_label:
            raise CuadDataError(f"repeated question for {label!s} in {title!r}")
        if bool(qa["is_impossible"]) != (not qa["answers"]):
            raise CuadDataError(f"is_impossible inconsistent with answers: {qa['id']!r}")
        spans_by_label[label] = sorted(
            (
                GoldSpan(
                    start=a["answer_start"], end=a["answer_start"] + len(a["text"]), text=a["text"]
                )
                for a in qa["answers"]
            ),
            key=lambda s: (s.start, s.end),
        )

    missing = [label for label in CuadLabel if label not in spans_by_label]
    if missing:
        raise CuadDataError(f"missing questions {missing} in {title!r}")

    annotations = []
    for label in CuadLabel:
        annotation = GoldAnnotation(
            contract_id=contract.contract_id, label=label, spans=tuple(spans_by_label[label])
        )
        check_offsets(annotation, contract)
        annotations.append(annotation)
    return contract, annotations


def _check_membership(all_titles: set[str], train: set[str], test: set[str]) -> None:
    if overlap := train & test:
        raise CuadDataError(f"{len(overlap)} titles in both train and test: {sorted(overlap)[:3]}")
    if unassigned := all_titles - train - test:
        raise CuadDataError(f"{len(unassigned)} titles in no split: {sorted(unassigned)[:3]}")
    if unknown := (train | test) - all_titles:
        raise CuadDataError(
            f"{len(unknown)} split titles not in {FULL_FILE}: {sorted(unknown)[:3]}"
        )
