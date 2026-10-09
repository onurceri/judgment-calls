"""Build the fixed, seeded evaluation samples from raw public datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from judgment_calls.datasets import loaders, sources
from judgment_calls.datasets.sampling import (
    DEFAULT_LENGTH_EDGES,
    bucket_label,
    length_bucket,
    stratified_sample,
)
from judgment_calls.datasets.store import write_items
from judgment_calls.models import Item

DEFAULT_SEED = 20261009


def _summeval_stratum(it: Item) -> int:
    vals = [float(v) for v in it.gold.values()]
    return round(sum(vals) / len(vals))


def prepare_task(task: str, raw_dir: Path, out_dir: Path, n: int, seed: int) -> dict[str, Any]:
    sources.download(task, raw_dir)
    extra: dict[str, Any] = {}
    if task == "faithfulness":
        pool = loaders.load_ragtruth(raw_dir)
        sample = stratified_sample(pool, n, lambda it: it.gold["hallucination"], seed)
    elif task == "summeval":
        pool = loaders.load_summeval(raw_dir)
        sample = stratified_sample(pool, n, _summeval_stratum, seed)
    elif task == "pairwise":
        pool, extra = loaders.load_mtbench(raw_dir)
        sample = stratified_sample(pool, n, lambda it: it.gold["preference"], seed)
    else:
        raise KeyError(task)
    write_items(out_dir / f"{task}.jsonl", sample)

    def dist(items: list[Item]) -> dict[str, Any]:
        if task == "summeval":
            labels = Counter(str(_summeval_stratum(it)) for it in items)
        else:
            labels = Counter(str(next(iter(it.gold.values()))) for it in items)
        buckets = Counter(
            bucket_label(length_bucket(int(it.meta["approx_tokens"])), DEFAULT_LENGTH_EDGES)
            for it in items
        )
        return {
            "n": len(items),
            "labels": dict(sorted(labels.items())),
            "length_buckets": dict(buckets),
        }

    return {"task": task, "seed": seed, "pool": dist(pool), "sample": dist(sample), **extra}


def prepare_all(
    tasks: list[str], raw_dir: Path, out_dir: Path, n: int, seed: int
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for task in tasks:
        manifest[task] = prepare_task(task, raw_dir, out_dir, n, seed)
    path = out_dir / "manifest.json"
    existing: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    existing.update(manifest)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return manifest
