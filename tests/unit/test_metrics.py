import pytest

from retrieval_labs.evaluation.extraction_metrics import (
    best_token_score,
    exact_match,
    normalize,
    token_score,
)
from retrieval_labs.evaluation.retrieval_metrics import (
    auroc,
    confusion,
    first_hit_rank,
    fit_threshold,
    hit_at_k,
    mean,
    reciprocal_rank,
    span_coverage,
)

# --- ranking ---


def test_first_hit_rank() -> None:
    assert first_hit_rank([False, False, True, True]) == 3
    assert first_hit_rank([False, False]) is None
    assert first_hit_rank([]) is None


def test_hit_at_k() -> None:
    assert hit_at_k(3, 3) and hit_at_k(1, 5)
    assert not hit_at_k(4, 3)
    assert not hit_at_k(None, 10)


def test_reciprocal_rank_with_cutoff() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(11, cutoff=10) == 0.0
    assert reciprocal_rank(None) == 0.0


def test_mrr_by_hand() -> None:
    # Items with first relevant ranks 1, 2, none, 4 -> (1 + 1/2 + 0 + 1/4) / 4
    assert mean([reciprocal_rank(r) for r in [1, 2, None, 4]]) == pytest.approx(1.75 / 4)


def test_span_coverage() -> None:
    per_rank = [[0], [], [1, 0], [2]]
    assert span_coverage(per_rank, n_spans=3, k=1) == pytest.approx(1 / 3)
    assert span_coverage(per_rank, n_spans=3, k=3) == pytest.approx(2 / 3)
    assert span_coverage(per_rank, n_spans=3, k=10) == 1.0
    with pytest.raises(ValueError):
        span_coverage([], n_spans=0, k=1)


# --- presence classification ---


def test_auroc_by_hand() -> None:
    assert auroc([3.0, 2.0], [1.0, 0.0]) == 1.0
    assert auroc([0.0], [1.0]) == 0.0
    # pairs: (2>1) 1, (2>3) 0, (1=1) 0.5, (1<3) 0 -> 1.5 / 4
    assert auroc([2.0, 1.0], [1.0, 3.0]) == pytest.approx(1.5 / 4)
    assert auroc([], [1.0]) is None


def test_confusion_and_f1() -> None:
    result = confusion([5, 4, 3, 2, 1], [True, False, True, False, False], threshold=3)
    assert (result.tp, result.fp, result.fn, result.tn) == (2, 1, 0, 2)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(0.8)


def test_confusion_with_no_predictions_has_zero_scores() -> None:
    result = confusion([1, 2], [True, False], threshold=10)
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


def test_fit_threshold_maximizes_f1() -> None:
    scores = [9.0, 8.0, 7.0, 2.0, 1.0]
    labels = [True, True, True, False, False]
    assert fit_threshold(scores, labels) == 7.0


def test_fit_threshold_prefers_lowest_on_ties() -> None:
    assert fit_threshold([2.0, 1.0], [True, True]) == 1.0


def test_fit_threshold_needs_positives() -> None:
    with pytest.raises(ValueError):
        fit_threshold([1.0], [False])


# --- extraction ---


def test_normalize() -> None:
    assert normalize("  The  Laws of the State, of “New York”. ") == "laws of state of new york"


def test_exact_match_against_any_gold() -> None:
    golds = ["governed by the laws of New York", "New York law applies."]
    assert exact_match("New York law applies", golds) == 1.0
    assert exact_match("Delaware law applies", golds) == 0.0
    assert exact_match(None, golds) == 0.0
    with pytest.raises(ValueError):
        exact_match("x", [])


def test_token_score_by_hand() -> None:
    # prediction tokens: laws of new york (4); gold: governed by laws of new york (6)
    score = token_score("the laws of New York", "governed by the laws of New York")
    assert score.precision == 1.0
    assert score.recall == pytest.approx(4 / 6)
    assert score.f1 == pytest.approx(2 * 1.0 * (4 / 6) / (1.0 + 4 / 6))


def test_token_score_counts_repeated_tokens_once_per_occurrence() -> None:
    score = token_score("days days days", "thirty days")
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(1 / 2)


def test_best_token_score_picks_best_gold() -> None:
    golds = ["unrelated sentence entirely", "liability shall not exceed fees paid"]
    assert best_token_score("liability shall not exceed fees paid", golds).f1 == 1.0
    assert best_token_score(None, golds).f1 == 0.0
