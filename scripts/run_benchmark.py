"""Retrieval benchmark on the prepared CUAD dataset (no LLM calls).

Examples:
    # Phase 3: BM25, chunk-size grid on dev
    uv run python scripts/run_benchmark.py --run-name phase3-bm25-dev \
        --chunking 50:17 --chunking 100:33 --chunking 150:50

    # Phase 4: the four methods on dev with 150/50 chunks
    uv run python scripts/run_benchmark.py --run-name phase4-dev \
        --method bm25 --method dense --method hybrid --method hybrid+rerank

Writes reports/benchmark/<run-name>/ (summary.json, items-*.jsonl, timings.json,
provenance.json, report.md) and failure samples with contract text to
data/analysis/<run-name>/ (git-ignored). The test split requires --final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import retrieval_labs
from retrieval_labs.clauses import ClauseType
from retrieval_labs.cuad.gold import GoldAnnotation
from retrieval_labs.cuad.labels import CuadLabel
from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import load_contracts, load_gold, read_manifest
from retrieval_labs.documents import Contract
from retrieval_labs.evaluation.bootstrap import (
    CONFIDENCE,
    N_RESAMPLES,
    SEED,
    bootstrap_ci,
    paired_difference_ci,
)
from retrieval_labs.evaluation.relevance import MIN_NORMALIZED_THRESHOLD, PRIMARY
from retrieval_labs.evaluation.retrieval_eval import (
    CUTOFFS,
    DEPTH,
    MRR_CUTOFF,
    WORD_BUDGET,
    ItemRecord,
    evaluate_split,
    latency_stats,
    per_contract_values,
    summarize_presence,
    summarize_ranking,
)
from retrieval_labs.ingestion.chunking import ChunkingConfig, chunk_contract
from retrieval_labs.retrieval.base import Retriever
from retrieval_labs.retrieval.bm25 import BM25Retriever
from retrieval_labs.retrieval.queries import EXPANDED_QUERIES, QUERIES

METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank", "bm25-expanded")
DENSE_BASED = ("dense", "hybrid", "hybrid+rerank")
# Dev-tuned rows (LOG-010): reported separately, never part of the fair comparison.
TUNED_ON_DEV = {"bm25-expanded"}
METHOD_QUERIES = {"bm25-expanded": EXPANDED_QUERIES}
BASELINE = "bm25"
CACHE_DIR = Path("data/cache/embeddings")
CI_METRICS = ("recall@1", "recall@5", f"mrr@{MRR_CUTOFF}")


def parse_chunking(value: str) -> ChunkingConfig:
    window, overlap = value.split(":")
    return ChunkingConfig(window_words=int(window), overlap_words=int(overlap))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def build_retrievers(methods: list[str]) -> tuple[dict[str, Retriever], dict[str, Any]]:
    """Instantiate the requested methods, loading each model once (model imports are lazy)."""
    retrievers: dict[str, Retriever] = {"bm25": BM25Retriever(), "bm25-expanded": BM25Retriever()}
    info: dict[str, Any] = {}
    if any(m in DENSE_BASED for m in methods):
        from retrieval_labs.retrieval.dense import DenseRetriever
        from retrieval_labs.retrieval.embeddings import CachedEmbedder, SentenceTransformerEmbedder
        from retrieval_labs.retrieval.hybrid import HybridRetriever

        started = time.perf_counter()
        model = SentenceTransformerEmbedder()
        info["embedder_load_s"] = round(time.perf_counter() - started, 3)
        info["embedder_max_seq_length"] = model.max_seq_length
        embedder = CachedEmbedder(model, CACHE_DIR, memoize_queries=False)
        info["embedder"] = embedder
        retrievers["dense"] = DenseRetriever(embedder)
        retrievers["hybrid"] = HybridRetriever([retrievers["bm25"], retrievers["dense"]])
    if "hybrid+rerank" in methods:
        from retrieval_labs.retrieval.reranker import (
            RerankingRetriever,
            SentenceTransformerCrossScorer,
        )

        started = time.perf_counter()
        scorer = SentenceTransformerCrossScorer()
        info["reranker_load_s"] = round(time.perf_counter() - started, 3)
        retrievers["hybrid+rerank"] = RerankingRetriever(retrievers["hybrid"], scorer)
    return {m: retrievers[m] for m in methods}, info


def queries_sha256(queries: Mapping[ClauseType, str]) -> str:
    payload = json.dumps({k.value: v for k, v in queries.items()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def describe(retriever: Retriever) -> dict[str, Any]:
    params = getattr(retriever, "params", None)
    return {"name": retriever.name, "params": params.model_dump() if params else {}}


def confidence_intervals(
    records: list[ItemRecord], baseline: list[ItemRecord] | None
) -> dict[str, Any]:
    values = per_contract_values(records)
    result: dict[str, Any] = {}
    for metric in CI_METRICS:
        ci = bootstrap_ci(values[metric])
        entry: dict[str, Any] = {"estimate": ci.estimate, "low": ci.low, "high": ci.high}
        if baseline is not None:
            diff = paired_difference_ci(values[metric], per_contract_values(baseline)[metric])
            entry["vs_baseline"] = {"estimate": diff.estimate, "low": diff.low, "high": diff.high}
        result[metric] = {k: _round_nested(v) for k, v in entry.items()}
    return result


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _round_nested(v) for k, v in value.items()}
    return round(value, 6) if isinstance(value, float) else value


def failure_samples(
    records: list[ItemRecord],
    contracts: list[Contract],
    gold: list[GoldAnnotation],
    config: ChunkingConfig,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Answerable items missed at k, with gold and top-3 chunk text for manual analysis."""
    by_id = {c.contract_id: c for c in contracts}
    spans = {(g.contract_id, g.label): g.spans for g in gold}
    samples = []
    for r in records:
        first = r.first_rank[PRIMARY]
        if not r.answerable or (first is not None and first <= k):
            continue
        contract = by_id[r.contract_id]
        chunk_text = {c.chunk_id: c.text for c in chunk_contract(contract, config)}
        samples.append(
            {
                "contract_id": r.contract_id,
                "title": contract.title,
                "label": r.label.value,
                "first_relevant_rank": first,
                "n_chunks": r.n_chunks,
                "gold": [s.text[:600] for s in spans[(r.contract_id, r.label)]],
                "top3": [
                    {"chunk_id": cid, "score": score, "text": chunk_text[cid][:600]}
                    for cid, score in zip(r.ranked_chunk_ids[:3], r.scores[:3], strict=True)
                ],
            }
        )
    return samples


def fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def fmt_ci(entry: dict[str, float]) -> str:
    return f"{entry['estimate']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"


def fmt_diff(entry: dict[str, float]) -> str:
    return f"{entry['estimate']:+.3f} [{entry['low']:+.3f}, {entry['high']:+.3f}]"


def render_report(summary: dict[str, Any], timings: dict[str, Any]) -> str:
    runs = summary["runs"]
    lines = [
        f"# Retrieval benchmark: {summary['run_name']}",
        "",
        f"- Split evaluated: **{summary['split']}**; presence thresholds fit on "
        f"**{summary['fit_split']}**",
        f"- Relevance: overlap / min(chunk, span) >= {MIN_NORMALIZED_THRESHOLD} (characters); "
        f"MRR cutoff {MRR_CUTOFF}; word budget {WORD_BUDGET}",
        f"- Bootstrap: {summary['bootstrap']['resamples']} contract-level resamples, seed "
        f"{summary['bootstrap']['seed']}, {int(summary['bootstrap']['confidence'] * 100)}% "
        "percentile intervals; differences are paired against the BM25 run with the same "
        "chunking",
        f"- Dataset archive SHA-256 `{summary['dataset']['archive_sha256'][:16]}...`",
        "",
        "Ranking metrics use answerable items only. Generated by scripts/run_benchmark.py.",
        "",
        "## Overview (micro average over answerable items)",
        "",
        "| Run | Items | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@750w | Span cov@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in runs.items():
        m = result["ranking"]["micro"]
        lines.append(
            f"| {name} | {m['answerable_items']} | "
            + " | ".join(fmt(m[f"recall@{k}"]) for k in CUTOFFS)
            + f" | {fmt(m[f'mrr@{MRR_CUTOFF}'])} | {fmt(m[f'recall@{WORD_BUDGET}words'])} "
            f"({m['results_in_word_budget']}) | {fmt(m['span_coverage@5'])} |"
        )
    lines += []
    if any(r.get("tuned_on_dev") for r in runs.values()):
        lines += [
            "",
            "† Tuned on dev (query expanded from the Phase 3 dev failure analysis, LOG-010). "
            "Not part of the fair comparison; on dev its numbers are optimistic by construction.",
        ]
    lines += [
        "",
        "## Confidence intervals (95%) and paired difference vs BM25",
        "",
        "| Run | R@1 | R@5 | MRR@10 | ΔR@1 vs BM25 | ΔR@5 vs BM25 | ΔMRR@10 vs BM25 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, result in runs.items():
        ci = result["confidence_intervals"]
        diffs = [
            fmt_diff(ci[m]["vs_baseline"]) if "vs_baseline" in ci[m] else "—" for m in CI_METRICS
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(fmt_ci(ci[m]) for m in CI_METRICS)
            + " | "
            + " | ".join(diffs)
            + " |"
        )
    lines += [
        "",
        "Macro averages (mean of per-label values):",
        "",
        "| Run | R@1 | R@5 | MRR@10 |",
        "|---|---:|---:|---:|",
    ]
    for name, result in runs.items():
        m = result["ranking"]["macro"]
        lines.append(
            f"| {name} | {fmt(m['recall@1'])} | {fmt(m['recall@5'])} | "
            f"{fmt(m[f'mrr@{MRR_CUTOFF}'])} |"
        )
    for name, result in runs.items():
        lines += [
            "",
            f"## {name}: per label",
            "",
            f"Retriever: `{result['retriever']['name']}` {result['retriever']['params']}",
            "",
            "| Label | Items | R@1 | R@5 | R@10 | MRR@10 | Span cov@5 | R@5 any-overlap "
            "| R@5 containment |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, m in result["ranking"]["per_label"].items():
            s = m["sensitivity"]
            lines.append(
                f"| {label} | {m['answerable_items']} | {fmt(m['recall@1'])} | "
                f"{fmt(m['recall@5'])} | {fmt(m['recall@10'])} | "
                f"{fmt(m[f'mrr@{MRR_CUTOFF}'])} | {fmt(m['span_coverage@5'])} | "
                f"{fmt(s['any_overlap']['recall@5'])} | {fmt(s['containment']['recall@5'])} |"
            )
        p = result["presence"]
        lines += [
            "",
            f"Presence detection (top-1 score >= threshold fit on {summary['fit_split']}):",
            "",
            "| Label | Present/Absent | AUROC | Threshold | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, m in p["per_label"].items():
            lines.append(
                f"| {label} | {m['present']}/{m['absent']} | {fmt(m['auroc'])} | "
                f"{fmt(m['threshold_fit_on_train'])} | {fmt(m['precision'])} | "
                f"{fmt(m['recall'])} | {fmt(m['f1'])} |"
            )
        pooled = p["pooled"]
        lines.append(
            f"| **pooled** | | macro {fmt(p['macro_auroc'])} | | {fmt(pooled['precision'])} | "
            f"{fmt(pooled['recall'])} | {fmt(pooled['f1'])} |"
        )
    lines += [
        "",
        "## Latency (this machine, CPU; see provenance.json)",
        "",
        "Index build includes embedding chunks for dense-based methods; cached embeddings make it",
        "faster, so cache hits/misses are listed. Query time includes encoding the query and,",
        "for reranking, scoring 20 candidates with the cross-encoder.",
        "",
        "| Run | Index build median ms | p95 ms | Query median ms | p95 ms | Embedding cache |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, t in timings["runs"].items():
        cache = t.get("embedding_cache", {})
        cache_text = f"{cache['hits']} hits / {cache['misses']} misses" if cache else "—"
        lines.append(
            f"| {name} | {t['index_build']['median_ms']} | {t['index_build']['p95_ms']} | "
            f"{t['query']['median_ms']} | {t['query']['p95_ms']} | {cache_text} |"
        )
    loads = {k: v for k, v in timings.get("models", {}).items() if k.endswith("_load_s")}
    if loads:
        lines += ["", "Model load times (s): " + ", ".join(f"{k} {v}" for k, v in loads.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=[s.value for s in Split], default=Split.DEV.value)
    parser.add_argument("--fit-split", choices=[s.value for s in Split], default=Split.TRAIN.value)
    parser.add_argument("--chunking", type=parse_chunking, action="append", default=[])
    parser.add_argument("--method", choices=METHODS, action="append", default=[])
    parser.add_argument("--final", action="store_true", help="allow the test split (once)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/benchmark"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    args = parser.parse_args()

    split, fit_split = Split(args.split), Split(args.fit_split)
    if Split.TEST in (split, fit_split) and not args.final:
        parser.error("the test split is reserved for the final evaluation; pass --final")
    if split == fit_split:
        parser.error("presence thresholds must be fit on a different split")
    configs = args.chunking or [ChunkingConfig()]
    methods = list(dict.fromkeys(args.method or [BASELINE]))
    if len(methods) > 1 and BASELINE not in methods:
        methods.insert(0, BASELINE)  # differences are reported against BM25

    provenance = {
        **git_provenance(),
        "package_version": retrieval_labs.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    manifest = read_manifest()
    retrievers, model_info = build_retrievers(methods)
    embedder = model_info.pop("embedder", None)
    contracts = load_contracts(split)
    gold = load_gold(split, final_evaluation=args.final)
    fit_contracts = load_contracts(fit_split)
    fit_gold = load_gold(fit_split, final_evaluation=args.final)

    run_dir = args.out_dir / args.run_name
    analysis_dir = args.analysis_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    run_timings: dict[str, Any] = {}
    for config in configs:
        baseline_records: list[ItemRecord] | None = None
        for method in methods:
            name = method if len(configs) == 1 else f"{method}/{config.label}"
            if len(methods) == 1:
                name = config.label  # Phase 3 naming: one method, several chunkings
            safe = name.replace("/", "_").replace("+", "-")
            if method in TUNED_ON_DEV:
                name += " †"
            print(f"evaluating {name} on {split.value} ...", flush=True)
            retriever = retrievers[method]
            queries = METHOD_QUERIES.get(method, QUERIES)
            hits_before = (embedder.hits, embedder.misses) if embedder else (0, 0)
            records, timing = evaluate_split(contracts, gold, retriever, config, queries)
            cache = (
                {"hits": embedder.hits - hits_before[0], "misses": embedder.misses - hits_before[1]}
                if embedder and method in DENSE_BASED
                else None
            )
            fit_records, _ = evaluate_split(fit_contracts, fit_gold, retriever, config, queries)
            if method == BASELINE:
                baseline_records = records
            compare = baseline_records if method != BASELINE else None
            runs[name] = {
                "method": method,
                "tuned_on_dev": method in TUNED_ON_DEV,
                "queries_sha256": queries_sha256(queries),
                "retriever": describe(retriever),
                "chunking": config.model_dump(),
                "ranking": summarize_ranking(records, config),
                "presence": summarize_presence(fit_records, records),
                "confidence_intervals": confidence_intervals(records, compare),
            }
            run_timings[name] = {
                "index_build": latency_stats(timing["index_build_s"]),
                "query": latency_stats(timing["query_s"]),
                **({"embedding_cache": cache} if cache else {}),
            }
            with (run_dir / f"items-{safe}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
                for record in records:
                    line = json.dumps(
                        record.model_dump(mode="json"), ensure_ascii=True, sort_keys=True
                    )
                    f.write(line + "\n")
            failures = failure_samples(records, contracts, gold, config)
            write_json(analysis_dir / f"failures-{safe}.json", failures)
            m = runs[name]["ranking"]["micro"]
            print(
                f"  R@1 {m['recall@1']:.3f}  R@5 {m['recall@5']:.3f}  "
                f"MRR@10 {m['mrr@10']:.3f}  misses@5 {len(failures)}",
                flush=True,
            )

    summary = {
        "run_name": args.run_name,
        "split": split.value,
        "fit_split": fit_split.value,
        "criterion": PRIMARY.value,
        "depth": DEPTH,
        "word_budget": WORD_BUDGET,
        "bootstrap": {"resamples": N_RESAMPLES, "seed": SEED, "confidence": CONFIDENCE},
        "queries_sha256": queries_sha256(QUERIES),
        "dataset": {
            "archive_sha256": manifest.archive_sha256,
            "files": {name: f.sha256 for name, f in manifest.files.items()},
        },
        "labels": [label.value for label in CuadLabel],
        "runs": runs,
    }
    timings = {"runs": run_timings, "models": model_info}
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "timings.json", timings)
    write_json(run_dir / "provenance.json", provenance)
    report = render_report(summary, timings)
    (run_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"written: {run_dir} (failure samples: {analysis_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
