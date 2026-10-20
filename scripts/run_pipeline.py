"""Run the contract analysis pipeline from the command line.

Examples:
    # Free dry run: real prompts, no model calls, worst-case cost for the paid model
    uv run python scripts/run_pipeline.py --dev-first 3 \
        --text-dir data/fixtures/adversarial --dry-run

    # Paid run (needs ANTHROPIC_API_KEY in .env and an explicit cap)
    uv run --env-file .env python scripts/run_pipeline.py --dev-first 3 \
        --text-dir data/fixtures/adversarial --provider anthropic --max-usd 0.25

    # Free dry run with the Phase 3 BM25 retriever (no model downloads)
    uv run python scripts/run_pipeline.py --dev-first 3 --dry-run --retriever bm25

Retrieval defaults to the method chosen after Phase 4 (hybrid + reranking). Contracts come
from the prepared dataset (train and dev splits only) or from UTF-8 text files. Reports are
written to data/runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import load_contracts
from retrieval_labs.documents import Contract
from retrieval_labs.extraction.budget import DEFAULT_LEDGER_PATH, BudgetExceededError, SpendLedger
from retrieval_labs.extraction.llm import (
    PRICES,
    Extractor,
    FakeExtractor,
    input_token_bound,
    worst_case_cost_usd,
)
from retrieval_labs.pipeline import ContractReport, PipelineConfig, analyze_contract
from retrieval_labs.retrieval.factory import APPLICATION_METHOD, METHODS, build_retriever

PAID_MODEL = "claude-haiku-5-5"
PAID_MAX_TOKENS = 4000


def contracts_from_args(args: argparse.Namespace) -> list[Contract]:
    contracts: list[Contract] = []
    if args.contract_id or args.dev_first:
        # The test split is never used here: it is reserved for final evaluation.
        available = {c.contract_id: c for s in (Split.TRAIN, Split.DEV) for c in load_contracts(s)}
        dev_ids = sorted(c.contract_id for c in load_contracts(Split.DEV))
        for contract_id in [*dev_ids[: args.dev_first], *args.contract_id]:
            if contract_id not in available:
                raise SystemExit(f"error: {contract_id} is not a train/dev contract")
            contracts.append(available[contract_id])
    text_files = [*args.text_file, *(p for d in args.text_dir for p in sorted(d.glob("*.txt")))]
    for path in text_files:
        text = path.read_bytes().decode("utf-8")  # no newline translation: offsets stay exact
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        contracts.append(Contract.from_text(f"file_{digest}", path.name, text))
    if not contracts:
        raise SystemExit("error: no contracts selected")
    return contracts


def make_extractor(args: argparse.Namespace) -> Extractor:
    if args.provider == "fake" or args.dry_run:
        return FakeExtractor()
    from retrieval_labs.extraction.anthropic_extractor import AnthropicExtractor

    return AnthropicExtractor(model=PAID_MODEL, effort="medium", max_tokens=PAID_MAX_TOKENS)


def print_dry_run(contracts: list[Contract], fake: FakeExtractor) -> None:
    price = PRICES[PAID_MODEL]
    rows = [
        (r, input_token_bound(r), worst_case_cost_usd(price, r, PAID_MAX_TOKENS))
        for r in fake.requests
    ]
    print(f"\nDry run: {len(contracts)} contract(s), {len(rows)} model call(s) would be made")
    print(f"{'clause':32} {'prompt chars':>12} {'input bound':>12} {'worst US$':>10}")
    for request, bound, cost in rows:
        chars = len(request.system) + len(request.user)
        print(f"{request.clause_type.value:32} {chars:12d} {bound:12d} {cost:10.5f}")
    total = sum(cost for _, _, cost in rows)
    print(f"Worst-case total for {PAID_MODEL} (max_tokens={PAID_MAX_TOKENS}): US$ {total:.4f}")


def print_report(report: ContractReport) -> None:
    print(f"\n== {report.title} ({report.contract_id}), {report.chunk_count} chunks")
    for section, results in report.by_section().items():
        print(f"  {section}")
        for result in results:
            text = (result.extracted_text or "").replace("\n", " ")
            snippet = f' "{text[:90]}{"..." if len(text) > 90 else ""}"' if text else ""
            print(f"    {result.clause_type.value:30} {result.status.value:10}{snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract-id", action="append", default=[])
    parser.add_argument("--dev-first", type=int, default=0, help="first N dev contracts by ID")
    parser.add_argument("--text-file", type=Path, action="append", default=[])
    parser.add_argument("--text-dir", type=Path, action="append", default=[], help="all *.txt")
    parser.add_argument("--provider", choices=["fake", "anthropic"], default="fake")
    parser.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    parser.add_argument("--max-usd", type=float, help="run spending cap (required for anthropic)")
    parser.add_argument("--retriever", choices=METHODS, default=APPLICATION_METHOD)
    parser.add_argument("--out-dir", type=Path, default=Path("data/runs"))
    args = parser.parse_args()

    paid = args.provider == "anthropic" and not args.dry_run
    if paid and args.max_usd is None:
        parser.error("--max-usd is required with --provider anthropic")

    contracts = contracts_from_args(args)
    extractor = make_extractor(args)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{extractor.provider}"
    ledger = SpendLedger(DEFAULT_LEDGER_PATH, run_id, args.max_usd) if paid else None
    config = PipelineConfig()
    retriever = build_retriever(args.retriever)

    reports: list[ContractReport] = []
    try:
        for contract in contracts:
            reports.append(analyze_contract(contract, retriever, extractor, config, run_id, ledger))
    except BudgetExceededError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
    finally:
        if ledger is not None:
            run_spent, project_spent = ledger.run_spent_usd, ledger.project_spent_usd
            print(f"\nrun spend US$ {run_spent:.5f}; project total US$ {project_spent:.5f}")

    if args.dry_run:
        print_dry_run(contracts, extractor)  # type: ignore[arg-type]
        return 0

    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        print_report(report)
        (out_dir / f"{report.contract_id}.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"\nreports written to {out_dir}")
    return 0 if len(reports) == len(contracts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
