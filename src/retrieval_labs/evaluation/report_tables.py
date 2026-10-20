"""Read stored benchmark runs (``reports/benchmark/<run>/``) into display rows.

The README and the demo both render their tables from here, from ``summary.json`` and
``timings-warm.json``, so no published number is ever retyped by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from retrieval_labs.clauses import ClauseType
from retrieval_labs.evaluation.labels import CUAD_LABEL_OF

REPORTS_DIR = Path("reports/benchmark")
TEST_RUN = "phase4-test"
# On dev the dev-tuned row was benchmarked in a separate run (Phase 4 step 8).
DEV_RUNS = ("phase4-dev", "phase4-dev-bm25-expanded")
METHOD_ORDER = ("bm25", "dense", "hybrid", "hybrid+rerank", "bm25-expanded")


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    low: float
    high: float

    def text(self, signed: bool = False) -> str:
        sign = "+" if signed else ""
        return f"{self.estimate:{sign}.3f} [{self.low:{sign}.3f}, {self.high:{sign}.3f}]"


class BenchmarkRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    tuned_on_dev: bool
    items: int
    recall_at_1: Interval
    recall_at_5: Interval
    mrr_at_10: Interval
    delta_recall_at_5: Interval | None  # paired vs BM25; None for BM25 itself
    presence_auroc: float
    query_median_ms: float
    query_p95_ms: float

    @property
    def label(self) -> str:
        return self.method + (" †" if self.tuned_on_dev else "")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(payload: dict[str, Any]) -> Interval:
    return Interval(estimate=payload["estimate"], low=payload["low"], high=payload["high"])


def warm_timings(run_dir: Path) -> dict[str, Any]:
    """Warm-cache timings: ``timings-warm.json`` from a repeated run, or the first run's
    ``timings.json`` when no method in it missed the embedding cache (e.g. BM25-only runs)."""
    warm = run_dir / "timings-warm.json"
    if warm.exists():
        return _read(warm)
    timings = _read(run_dir / "timings.json")
    missed = [
        name
        for name, run in timings["runs"].items()
        if run.get("embedding_cache", {}).get("misses", 0) > 0
    ]
    if missed:
        raise ValueError(f"{run_dir.name}: first-run timings include cache misses for {missed}")
    return timings


def load_rows(
    run_names: tuple[str, ...] | str, reports_dir: Path = REPORTS_DIR
) -> list[BenchmarkRow]:
    """One row per method across the given runs, in ``METHOD_ORDER``; the first run wins."""
    names = (run_names,) if isinstance(run_names, str) else run_names
    rows: dict[str, BenchmarkRow] = {}
    for name in names:
        summary = _read(reports_dir / name / "summary.json")
        timings = warm_timings(reports_dir / name)
        for key, run in summary["runs"].items():
            if run["method"] in rows:
                continue
            ci = run["confidence_intervals"]
            delta = ci["recall@5"].get("vs_baseline")
            query = timings["runs"][key]["query"]
            rows[run["method"]] = BenchmarkRow(
                method=run["method"],
                tuned_on_dev=run["tuned_on_dev"],
                items=run["ranking"]["micro"]["answerable_items"],
                recall_at_1=_interval(ci["recall@1"]),
                recall_at_5=_interval(ci["recall@5"]),
                mrr_at_10=_interval(ci["mrr@10"]),
                delta_recall_at_5=_interval(delta) if delta else None,
                presence_auroc=run["presence"]["macro_auroc"],
                query_median_ms=query["median_ms"],
                query_p95_ms=query["p95_ms"],
            )
    return [rows[m] for m in METHOD_ORDER if m in rows]


TABLE_HEADER = (
    "| Method | R@1 | R@5 | MRR@10 | ΔR@5 vs BM25 | Presence AUROC | Query median (p95) |\n"
    "|---|---|---|---|---|---:|---:|"
)


def markdown_table(rows: list[BenchmarkRow]) -> str:
    """The results table exactly as the README shows it (a test keeps the two in sync)."""
    lines = [TABLE_HEADER]
    for row in rows:
        delta = row.delta_recall_at_5.text(signed=True) if row.delta_recall_at_5 else "—"
        lines.append(
            f"| {row.label} | {row.recall_at_1.text()} | {row.recall_at_5.text()} | "
            f"{row.mrr_at_10.text()} | {delta} | {row.presence_auroc:.3f} | "
            f"{row.query_median_ms:.2f} ms ({row.query_p95_ms:.2f}) |"
        )
    return "\n".join(lines)


def presence_thresholds(
    method: str, run_name: str = TEST_RUN, reports_dir: Path = REPORTS_DIR
) -> dict[ClauseType, float]:
    """Top-1 score thresholds fit on train for one method (stored with every run)."""
    summary = _read(reports_dir / run_name / "summary.json")
    run = next(r for r in summary["runs"].values() if r["method"] == method)
    per_label = run["presence"]["per_label"]
    return {
        clause: per_label[label.value]["threshold_fit_on_train"]
        for clause, label in CUAD_LABEL_OF.items()
    }
