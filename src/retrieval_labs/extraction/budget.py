"""Spend ledger and budget guard for paid LLM calls.

Every paid call is checked *before* it is made against its worst-case cost, and
recorded afterwards with its measured cost. The ledger is an append-only JSONL file
that persists across runs, so the project-wide budget is enforced, not just per run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from retrieval_labs.extraction.schema import LLMUsage

PROJECT_BUDGET_USD = 20.0  # brief Addendum 21.4
DEFAULT_LEDGER_PATH = Path("data/llm_ledger.jsonl")


class BudgetExceededError(RuntimeError):
    pass


class SpendLedger:
    def __init__(
        self,
        path: Path,
        run_id: str,
        run_cap_usd: float,
        project_cap_usd: float = PROJECT_BUDGET_USD,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.run_cap_usd = run_cap_usd
        self.project_cap_usd = project_cap_usd
        self.run_spent_usd = 0.0
        self.project_spent_usd = sum(entry["cost_usd"] for entry in self._entries())

    def _entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        lines = self.path.read_bytes().decode("utf-8").split("\n")
        return [json.loads(line) for line in lines if line]

    def check(self, worst_case_usd: float) -> None:
        """Raise unless the next call fits in both budgets even in its worst case."""
        if self.run_spent_usd + worst_case_usd > self.run_cap_usd:
            raise BudgetExceededError(
                f"run cap US$ {self.run_cap_usd:.4f} would be exceeded: spent "
                f"{self.run_spent_usd:.4f} + worst case {worst_case_usd:.4f}"
            )
        if self.project_spent_usd + worst_case_usd > self.project_cap_usd:
            raise BudgetExceededError(
                f"project budget US$ {self.project_cap_usd:.2f} would be exceeded: spent "
                f"{self.project_spent_usd:.4f} + worst case {worst_case_usd:.4f}"
            )

    def record(self, *, contract_id: str, clause_type: str, model: str, usage: LLMUsage) -> None:
        entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "contract_id": contract_id,
            "clause_type": clause_type,
            "model": model,
            **usage.model_dump(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        self.run_spent_usd += usage.cost_usd
        self.project_spent_usd += usage.cost_usd
