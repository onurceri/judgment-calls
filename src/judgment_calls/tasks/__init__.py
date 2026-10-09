"""Judge task specifications.

Each task exposes a versioned :class:`TaskSpec` (the single source of truth for the
wording shown to *both* judge families) and a ``variants`` function that turns a
dataset item into the concrete presentations that get judged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from judgment_calls.models import Item, TaskSpec, Variant
from judgment_calls.tasks import faithfulness, pairwise, summeval


@dataclass(frozen=True)
class TaskDef:
    spec: TaskSpec
    variants: Callable[[Item], list[Variant]]


TASKS: dict[str, TaskDef] = {
    faithfulness.SPEC.task: TaskDef(faithfulness.SPEC, faithfulness.variants),
    summeval.SPEC.task: TaskDef(summeval.SPEC, summeval.variants),
    pairwise.SPEC.task: TaskDef(pairwise.SPEC, pairwise.variants),
}


def get_task(name: str) -> TaskDef:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}") from exc
