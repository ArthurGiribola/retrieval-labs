"""End-to-end check of the running stack: submit a contract, poll its job, verify the report.

    uv run python scripts/api_smoke.py                  # a CUAD test-split contract (text only)
    uv run python scripts/api_smoke.py --file my.txt    # any UTF-8 .txt
    uv run python scripts/api_smoke.py --keep           # do not delete it afterwards

Needs the API on ``--base-url`` (default http://127.0.0.1:8000) and a worker. Only contract
text is sent; gold annotations are never loaded. Exits non-zero on any failed check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

from retrieval_labs.clauses import ClauseType
from retrieval_labs.cuad.splits import Split
from retrieval_labs.cuad.store import load_contracts


def pick_text(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.file:
        path = Path(args.file)
        return path.name, path.stem, path.read_bytes().decode("utf-8")
    contracts = load_contracts(Split.TEST)
    chosen = next((c for c in contracts if c.contract_id == args.contract_id), None)
    by_length = sorted(contracts, key=lambda c: (len(c.text), c.contract_id))
    chosen = chosen or by_length[len(by_length) // 2]
    return f"{chosen.contract_id}.txt", chosen.title, chosen.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", help="a UTF-8 .txt contract to submit")
    parser.add_argument("--contract-id", help="CUAD test contract id (default: median length)")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--keep", action="store_true", help="keep the contract afterwards")
    args = parser.parse_args()

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(("ok    " if condition else "FAIL  ") + message)
        if not condition:
            failures.append(message)

    filename, title, text = pick_text(args)
    print(f"submitting {filename!r} ({len(text):,} characters)")
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        health = client.get("/health").json()
        check(health.get("status") == "ok", f"health: {health}")
        started = time.monotonic()
        response = client.post(
            "/contracts",
            files={"file": (filename, text.encode("utf-8"), "text/plain")},
            data={"title": title[:255]},
        )
        check(response.status_code == 202, f"submit answered {response.status_code}")
        if response.status_code != 202:
            return 1
        submission = response.json()

        status: dict = {}
        while time.monotonic() - started < args.timeout_s:
            status = client.get(submission["status_url"]).json()
            if status["state"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(1)
        elapsed = time.monotonic() - started
        check(
            status.get("state") == "COMPLETED", f"job {status.get('state')} after {elapsed:.1f} s"
        )
        if status.get("state") != "COMPLETED":
            print(f"error_code: {status.get('error_code')}")
            return 1

        report = client.get(submission["report_url"]).json()["report"]
        results = report["results"]
        check(report["contract_id"] == submission["document_id"], "report is for this contract")
        check(
            {r["clause_type"] for r in results} == {c.value for c in ClauseType},
            f"one result per clause type ({len(results)})",
        )
        check(report["chunk_count"] > 0, f"{report['chunk_count']} chunks indexed")
        top_k = report["run"]["top_k"]
        check(
            all(
                len(r["retrieved_chunk_ids"]) == min(top_k, report["chunk_count"]) for r in results
            ),
            f"top {top_k} passages retrieved for every clause",
        )
        spans = [e for r in results for e in r["evidence"]]
        check(
            all(text[e["source_start"] : e["source_end"]] == e["exact_quote"] for e in spans),
            f"{len(spans)} evidence spans match the uploaded text exactly",
        )
        run = report["run"]
        print(
            f"retriever: {run['retriever']}  provider: {run['provider']}",
            f"prompt: {run['prompt_version']}",
        )
        print("timings_s:", report["timings_s"])
        for result in results:
            print(f"  {result['clause_type']:<28} {result['status']}")

        if not args.keep:
            check(
                client.delete(f"/contracts/{submission['document_id']}").status_code == 204,
                "deleted",
            )

    print("PASSED" if not failures else f"FAILED: {len(failures)} check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
