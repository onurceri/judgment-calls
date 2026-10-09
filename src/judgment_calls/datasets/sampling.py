"""Deterministic stratified sampling with a length spread."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence

from judgment_calls.models import Item

DEFAULT_LENGTH_EDGES: tuple[int, ...] = (700, 1400, 2100)


def approx_tokens(state: dict[str, str]) -> int:
    """Cheap, model-agnostic token estimate (~4 characters per token)."""
    return max(1, round(sum(len(v) for v in state.values()) / 4))


def length_bucket(tokens: int, edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> int:
    for i, edge in enumerate(edges):
        if tokens < edge:
            return i
    return len(edges)


def bucket_label(bucket: int, edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> str:
    lo = 0 if bucket == 0 else edges[bucket - 1]
    if bucket >= len(edges):
        return f">={lo}"
    return f"{lo}-{edges[bucket]}"


def allocate(target: int, capacities: Sequence[int]) -> list[int]:
    """Spread `target` as evenly as possible over bins, never exceeding a bin's capacity.

    Water-filling: bins that cannot take their equal share give the remainder to the
    others. Remainders go to the earliest bins with spare capacity, deterministically.
    """
    alloc = [0] * len(capacities)
    remaining = min(target, sum(capacities))
    open_bins = [i for i, c in enumerate(capacities) if c > 0]
    while remaining > 0 and open_bins:
        share, extra = divmod(remaining, len(open_bins))
        next_open: list[int] = []
        given = 0
        for rank, i in enumerate(open_bins):
            want = share + (1 if rank < extra else 0)
            take = min(want, capacities[i] - alloc[i])
            alloc[i] += take
            given += take
            if alloc[i] < capacities[i]:
                next_open.append(i)
        remaining -= given
        if given == 0:
            break
        open_bins = next_open
    return alloc


def stratified_sample(
    items: Sequence[Item],
    n: int,
    stratum: Callable[[Item], Hashable],
    seed: int,
    edges: Sequence[int] = DEFAULT_LENGTH_EDGES,
) -> list[Item]:
    """Sample ~n items: equal share per label stratum, then equal share per length bucket."""
    rng = random.Random(seed)
    by_stratum: dict[Hashable, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda x: x.item_id):
        by_stratum[stratum(it)].append(it)
    strata = sorted(by_stratum, key=str)
    per_stratum = allocate(n, [len(by_stratum[s]) for s in strata])
    chosen: list[Item] = []
    for s, k in zip(strata, per_stratum, strict=True):
        buckets: dict[int, list[Item]] = defaultdict(list)
        for it in by_stratum[s]:
            buckets[length_bucket(approx_tokens(it.state), edges)].append(it)
        keys = sorted(buckets)
        per_bucket = allocate(k, [len(buckets[b]) for b in keys])
        for b, kb in zip(keys, per_bucket, strict=True):
            pool = list(buckets[b])
            rng.shuffle(pool)
            chosen.extend(pool[:kb])
    chosen.sort(key=lambda x: x.item_id)
    return chosen
