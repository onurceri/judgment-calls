"""Reading/writing sampled items (data/samples/<task>.jsonl)."""

from __future__ import annotations

from pathlib import Path

from judgment_calls.models import Item


def write_items(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


def read_items(path: Path) -> list[Item]:
    with path.open() as fh:
        return [Item.model_validate_json(line) for line in fh if line.strip()]
