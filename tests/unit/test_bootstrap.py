import random

import pytest

from retrieval_labs.evaluation.bootstrap import bootstrap_ci, paired_difference_ci

MIXED = {"c1": [1.0, 0.0], "c2": [1.0], "c3": [0.0, 0.0, 1.0], "c4": [1.0, 1.0]}


def random_hits(n_contracts: int, p: float, seed: int = 0) -> dict[str, list[float]]:
    """Contracts with 1-5 items each, hits drawn with probability p (varied contract means)."""
    rng = random.Random(seed)
    return {
        f"c{i:03d}": [float(rng.random() < p) for _ in range(rng.randint(1, 5))]
        for i in range(n_contracts)
    }


def test_estimate_is_the_plain_item_mean() -> None:
    interval = bootstrap_ci(MIXED)
    assert interval.estimate == pytest.approx(5 / 8)
    assert interval.low <= interval.estimate <= interval.high


def test_constant_values_give_a_degenerate_interval() -> None:
    interval = bootstrap_ci({"c1": [1.0, 1.0], "c2": [1.0]})
    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)


def test_deterministic_for_a_fixed_seed() -> None:
    values = random_hits(100, 0.6)
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    assert bootstrap_ci(values, seed=7) != bootstrap_ci(values, seed=8)


def test_contracts_without_items_are_ignored() -> None:
    assert bootstrap_ci({**MIXED, "empty": []}) == bootstrap_ci(MIXED)


def test_identical_methods_have_zero_difference() -> None:
    interval = paired_difference_ci(MIXED, MIXED)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_consistently_better_method_has_positive_interval() -> None:
    baseline = random_hits(100, 0.6)
    better = {c: [1.0] * len(v) for c, v in baseline.items()}
    interval = paired_difference_ci(better, baseline)
    n_items = sum(len(v) for v in baseline.values())
    misses = sum(1 for v in baseline.values() for x in v if x == 0.0)
    assert interval.estimate == pytest.approx(misses / n_items)
    assert interval.low > 0


def test_tiny_samples_cannot_exclude_zero() -> None:
    """With 4 contracts, 2 of them ties, resamples of only ties are likely: low bound is 0."""
    better = {c: [1.0] * len(v) for c, v in MIXED.items()}
    assert paired_difference_ci(better, MIXED).low == 0.0


def test_pairing_requires_same_items() -> None:
    with pytest.raises(ValueError, match="same items"):
        paired_difference_ci(MIXED, {"c1": [1.0]})


def test_wider_interval_with_fewer_contracts() -> None:
    small = random_hits(10, 0.6, seed=1)
    large = random_hits(400, 0.6, seed=1)
    width = lambda i: i.high - i.low  # noqa: E731
    assert width(bootstrap_ci(large)) < width(bootstrap_ci(small))
