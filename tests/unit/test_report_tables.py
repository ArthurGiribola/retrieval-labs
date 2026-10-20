import json
from pathlib import Path
from typing import Any

import pytest

from retrieval_labs.clauses import ClauseType
from retrieval_labs.evaluation.labels import CUAD_LABEL_OF
from retrieval_labs.evaluation.report_tables import (
    DEV_RUNS,
    TEST_RUN,
    load_rows,
    markdown_table,
    presence_thresholds,
    warm_timings,
)

ROOT = Path(__file__).resolve().parents[2]
STORED_RUNS = ROOT / "reports" / "benchmark"


def interval(estimate: float, low: float, high: float, delta: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {"estimate": estimate, "low": low, "high": high}
    if delta:
        payload["vs_baseline"] = {"estimate": -0.05, "low": -0.1, "high": 0.0}
    return payload


def run(method: str, tuned: bool = False, auroc: float = 0.8) -> dict[str, Any]:
    baseline = method == "bm25"
    return {
        "method": method,
        "tuned_on_dev": tuned,
        "ranking": {"micro": {"answerable_items": 10}},
        "confidence_intervals": {
            metric: interval(0.5, 0.4, 0.6, delta=not baseline)
            for metric in ("recall@1", "recall@5", "mrr@10")
        },
        "presence": {
            "macro_auroc": auroc,
            "per_label": {
                label.value: {"threshold_fit_on_train": float(i)}
                for i, label in enumerate(CUAD_LABEL_OF.values())
            },
        },
    }


def timing(median: float, misses: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"query": {"median_ms": median, "p95_ms": median * 2}}
    if misses is not None:
        entry["embedding_cache"] = {"hits": 1, "misses": misses}
    return entry


def write_run(
    reports: Path,
    name: str,
    runs: dict[str, dict[str, Any]],
    timings: dict[str, Any],
    warm: dict[str, Any] | None = None,
) -> None:
    directory = reports / name
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    (directory / "timings.json").write_text(json.dumps({"runs": timings}), encoding="utf-8")
    if warm is not None:
        (directory / "timings-warm.json").write_text(json.dumps({"runs": warm}), encoding="utf-8")


def test_rows_follow_method_order_and_first_run_wins(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "main",
        {"dense": run("dense"), "bm25": run("bm25", auroc=0.9)},
        {"dense": timing(99.0, misses=3), "bm25": timing(9.0)},
        warm={"dense": timing(15.0, misses=0), "bm25": timing(0.2)},
    )
    write_run(
        tmp_path,
        "extra",
        {"bm25": run("bm25", auroc=0.1), "bm25-expanded †": run("bm25-expanded", tuned=True)},
        {"bm25": timing(0.3), "bm25-expanded †": timing(0.4)},
    )
    rows = load_rows(("main", "extra"), tmp_path)
    assert [r.method for r in rows] == ["bm25", "dense", "bm25-expanded"]
    assert rows[0].presence_auroc == 0.9  # from "main", not "extra"
    assert rows[0].delta_recall_at_5 is None
    assert rows[1].query_median_ms == 15.0  # warm timings, not the first run's 99 ms
    assert rows[2].label == "bm25-expanded †"
    assert rows[2].query_median_ms == 0.4  # BM25-only run: first-run timings are warm


def test_first_run_timings_with_cache_misses_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "cold", {"dense": run("dense")}, {"dense": timing(99.0, misses=3)})
    with pytest.raises(ValueError, match="cache misses"):
        warm_timings(tmp_path / "cold")


def test_markdown_table_formats_intervals_and_signed_deltas(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "main",
        {"bm25": run("bm25"), "dense": run("dense")},
        {"bm25": timing(0.25), "dense": timing(16.0)},
    )
    lines = markdown_table(load_rows("main", tmp_path)).split("\n")
    assert lines[2] == (
        "| bm25 | 0.500 [0.400, 0.600] | 0.500 [0.400, 0.600] | 0.500 [0.400, 0.600] | — "
        "| 0.800 | 0.25 ms (0.50) |"
    )
    assert "| -0.050 [-0.100, +0.000] |" in lines[3]


def test_presence_thresholds_map_every_clause_type(tmp_path: Path) -> None:
    write_run(tmp_path, "main", {"bm25": run("bm25")}, {"bm25": timing(0.2)})
    thresholds = presence_thresholds("bm25", "main", tmp_path)
    assert set(thresholds) == set(ClauseType)
    assert thresholds[ClauseType.GOVERNING_LAW] == 0.0


def test_stored_phase4_runs_load_with_all_five_methods() -> None:
    methods = ["bm25", "dense", "hybrid", "hybrid+rerank", "bm25-expanded"]
    for runs in (TEST_RUN, DEV_RUNS):
        rows = load_rows(runs, STORED_RUNS)
        assert [r.method for r in rows] == methods
        assert [r.tuned_on_dev for r in rows] == [False] * 4 + [True]
    test = {r.method: r for r in load_rows(TEST_RUN, STORED_RUNS)}
    summary = json.loads((STORED_RUNS / TEST_RUN / "summary.json").read_text(encoding="utf-8"))
    stored = summary["runs"]["hybrid+rerank"]
    assert test["hybrid+rerank"].recall_at_5.estimate == stored["ranking"]["micro"]["recall@5"]
    assert test["hybrid+rerank"].presence_auroc == stored["presence"]["macro_auroc"]
