"""Hard spend cap enforced in code.

Every paid call is recorded in an append-only ledger (actual `usage.cost`). Before a
call is sent, its worst-case cost is *reserved*; if spent + reserved + estimate would
exceed the cap, :class:`BudgetExceeded` is raised and nothing is sent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cap_usd: float, ledger_path: Path) -> None:
        self.cap_usd = cap_usd
        self.ledger_path = ledger_path
        self.reserved_usd = 0.0
        self.spent_usd = self._load()

    def _load(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        total = 0.0
        with self.ledger_path.open() as fh:
            for line in fh:
                if line.strip():
                    total += float(json.loads(line)["cost_usd"])
        return total

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd - self.reserved_usd

    def reserve(self, estimate_usd: float) -> None:
        if self.spent_usd + self.reserved_usd + estimate_usd > self.cap_usd:
            raise BudgetExceeded(
                f"cap ${self.cap_usd:.4f} would be exceeded "
                f"(spent ${self.spent_usd:.4f}, reserved ${self.reserved_usd:.4f}, "
                f"next ~${estimate_usd:.5f})"
            )
        self.reserved_usd += estimate_usd

    def settle(self, estimate_usd: float, actual_usd: float | None, label: str) -> float:
        """Release the reservation and book the actual cost (estimate if unknown)."""
        self.reserved_usd = max(0.0, self.reserved_usd - estimate_usd)
        cost = estimate_usd if actual_usd is None else actual_usd
        self.spent_usd += cost
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "label": label,
                        "cost_usd": cost,
                        "estimated": actual_usd is None,
                    }
                )
                + "\n"
            )
        return cost

    def release(self, estimate_usd: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - estimate_usd)
