"""Contract-level split assignment: official test, plus a dev split carved from official train."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Set
from enum import StrEnum

from retrieval_labs.cuad import CuadDataError
from retrieval_labs.documents import Contract

DEV_SIZE = 102  # same size as the official test split
DEV_SEED = 42


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


def _text_groups(contracts: Iterable[Contract]) -> list[list[str]]:
    """All contract IDs grouped by identical text; groups and members sorted."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for contract in contracts:
        by_hash[contract.text_sha256].append(contract.contract_id)
    return sorted(sorted(ids) for ids in by_hash.values())


def duplicate_text_groups(contracts: Iterable[Contract]) -> list[list[str]]:
    """Groups of contract IDs whose texts are byte-identical (groups of size > 1 only)."""
    return [ids for ids in _text_groups(contracts) if len(ids) > 1]


def assign_splits(
    contracts: Iterable[Contract],
    official_train_ids: Set[str],
    official_test_ids: Set[str],
    dev_size: int = DEV_SIZE,
    seed: int = DEV_SEED,
) -> dict[str, Split]:
    """Map every contract ID to a split.

    The official test split is kept as-is. The dev split is sampled from official train
    by *document identity*: contracts with identical text form one group and always land
    in the same split, so a duplicate cannot leak between train and dev. The result
    depends only on the contract IDs, texts and seed, not on input order.
    """
    contracts = list(contracts)
    groups = _text_groups(contracts)
    for ids in groups:
        in_train = [i for i in ids if i in official_train_ids]
        if in_train and len(in_train) != len(ids):
            raise CuadDataError(f"identical text across official splits: {ids}")

    train_groups = [ids for ids in groups if ids[0] in official_train_ids]
    random.Random(seed).shuffle(train_groups)
    dev: list[str] = []
    for ids in train_groups:
        if len(dev) + len(ids) <= dev_size:
            dev.extend(ids)
        if len(dev) == dev_size:
            break
    if len(dev) != dev_size:
        raise CuadDataError(f"cannot draw {dev_size} dev contracts from official train")

    dev_ids = set(dev)
    assignment: dict[str, Split] = {}
    for contract in contracts:
        cid = contract.contract_id
        if cid in official_test_ids:
            assignment[cid] = Split.TEST
        elif cid in official_train_ids:
            assignment[cid] = Split.DEV if cid in dev_ids else Split.TRAIN
        else:
            raise CuadDataError(f"contract in no official split: {cid}")
    return dict(sorted(assignment.items()))
