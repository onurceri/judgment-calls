"""Load the append-only raw logs into de-duplicated judgment tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from judgment_calls.datasets.store import read_items
from judgment_calls.models import Item, JudgmentRecord

Key = tuple[str, str, str, int]  # judge_id, item_id, variant, repeat


@dataclass
class TaskResults:
    task: str
    items: dict[str, Item]
    records: dict[Key, JudgmentRecord]
    errors: list[JudgmentRecord]

    def judges(self) -> list[str]:
        return sorted({k[0] for k in self.records})

    def for_judge(self, judge_id: str, repeat: int = 0) -> dict[tuple[str, str], JudgmentRecord]:
        return {
            (k[1], k[2]): r for k, r in self.records.items() if k[0] == judge_id and k[3] == repeat
        }


def load_task(task: str, results_dir: Path, samples_dir: Path) -> TaskResults:
    items = {it.item_id: it for it in read_items(samples_dir / f"{task}.jsonl")}
    records: dict[Key, JudgmentRecord] = {}
    errors: list[JudgmentRecord] = []
    path = results_dir / "raw" / f"{task}.jsonl"
    if path.exists():
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = JudgmentRecord.model_validate_json(line)
                if rec.item_id not in items:
                    continue  # item no longer in the current sample
                if rec.error is not None:
                    errors.append(rec)
                    continue
                # later successful records supersede earlier ones (e.g. config fixes)
                records[(rec.judge_id, rec.item_id, rec.variant, rec.repeat)] = rec
    successful = {(k[0], k[1], k[2], k[3]) for k in records}
    unresolved = [
        e for e in errors if (e.judge_id, e.item_id, e.variant, e.repeat) not in successful
    ]
    return TaskResults(task, items, records, unresolved)
