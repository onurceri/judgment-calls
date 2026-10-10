"""Run judges over task items; append results to JSONL with full provenance."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from judgment_calls.budget import Budget, BudgetExceeded
from judgment_calls.cache import ResponseCache, request_hash
from judgment_calls.datasets.store import read_items
from judgment_calls.http import ProviderError, RetryPolicy, post_json
from judgment_calls.models import Item, JudgeConfig, JudgmentRecord, sha256_hex
from judgment_calls.providers import Judge, make_judge
from judgment_calls.providers.base import ParseError
from judgment_calls.tasks import get_task


def hash_order(items: list[Item]) -> list[Item]:
    """Deterministic pseudo-random order; prefixes are nested subsets (5 ⊂ 40 ⊂ 200)."""
    return sorted(items, key=lambda it: sha256_hex(it.item_id))


def read_subset(path: Path) -> set[str]:
    """Item ids, one per line ('#' starts a comment)."""
    lines = (ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines())
    return {ln for ln in lines if ln}


@dataclass
class RunConfig:
    tasks: list[str]
    judges: list[JudgeConfig]
    samples_dir: Path
    results_dir: Path
    cache_path: Path
    budget_cap_usd: float
    limit: int | None = None
    repeats: int = 1
    repeat_limit: int | None = None  # repeats > 1 only on the first N hash-ordered items
    concurrency: int = 16
    per_judge_concurrency: int = 6
    timeout_s: float = 120.0
    dry_run: bool = False
    subset_dir: Path | None = None  # if set, only items listed in <subset_dir>/<task>.txt


@dataclass
class Job:
    task: str
    item: Item
    variant_name: str
    state: dict[str, str]
    gold: dict[str, bool | str | float]
    judge: Judge
    repeat: int
    rank: int = 0  # position of the item in hash order (used to interleave tasks)


@dataclass
class RunSummary:
    run_id: str
    planned: int = 0
    skipped_done: int = 0
    cached: int = 0
    paid: int = 0
    errors: int = 0
    spent_usd: float = 0.0
    estimated_usd: float = 0.0
    budget_stopped: bool = False
    by_judge_cost: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def _done_keys(path: Path) -> set[tuple[str, str, str, int, str]]:
    keys: set[tuple[str, str, str, int, str]] = set()
    if not path.exists():
        return keys
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is None:
                keys.add(
                    (r["judge_id"], r["item_id"], r["variant"], r["repeat"], r["request_hash"])
                )
    return keys


def plan_jobs(cfg: RunConfig) -> list[Job]:
    jobs: list[Job] = []
    judges = [make_judge(j) for j in cfg.judges]
    for task in cfg.tasks:
        tdef = get_task(task)
        items = hash_order(read_items(cfg.samples_dir / f"{task}.jsonl"))
        if cfg.subset_dir is not None:
            wanted = read_subset(cfg.subset_dir / f"{task}.txt")
            items = [it for it in items if it.item_id in wanted]
        if cfg.limit is not None:
            items = items[: cfg.limit]
        repeat_ids = {
            it.item_id for it in (items if cfg.repeat_limit is None else items[: cfg.repeat_limit])
        }
        # Item-major order: a budget stop leaves every judge on the same item prefix,
        # which keeps paired comparisons valid.
        for rank, it in enumerate(items):
            n_rep = cfg.repeats if it.item_id in repeat_ids else 1
            for variant in tdef.variants(it):
                for rep in range(n_rep):
                    for judge in judges:
                        jobs.append(
                            Job(
                                task,
                                it,
                                variant.name,
                                variant.state,
                                variant.gold,
                                judge,
                                rep,
                                rank,
                            )
                        )
    # Interleave tasks by item rank (stable sort keeps the within-item order).
    jobs.sort(key=lambda j: j.rank)
    return jobs


async def run(cfg: RunConfig) -> RunSummary:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    summary = RunSummary(run_id=run_id)
    raw_dir = cfg.results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(cfg.budget_cap_usd, cfg.results_dir / "spend_ledger.jsonl")
    cache = ResponseCache(cfg.cache_path)
    jobs = plan_jobs(cfg)
    summary.planned = len(jobs)

    done: dict[str, set[tuple[str, str, str, int, str]]] = {
        t: _done_keys(raw_dir / f"{t}.jsonl") for t in cfg.tasks
    }
    pending: list[Job] = []
    for job in jobs:
        spec = get_task(job.task).spec
        rhash = request_hash(
            job.judge.endpoint, job.judge.build_request(spec, job.state), job.repeat
        )
        key = (job.judge.config.id, job.item.item_id, job.variant_name, job.repeat, rhash)
        if key in done[job.task]:
            summary.skipped_done += 1
        else:
            pending.append(job)

    if cfg.dry_run:
        for job in pending:
            spec = get_task(job.task).spec
            body = job.judge.build_request(spec, job.state)
            if cache.get(request_hash(job.judge.endpoint, body, job.repeat)) is None:
                summary.estimated_usd += job.judge.estimate_cost(body)
        cache.close()
        return summary

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/onurceri/judgment-calls",
        "X-Title": "judgment-calls",
    }
    global_sem = asyncio.Semaphore(cfg.concurrency)
    judge_sems = {j.id: asyncio.Semaphore(cfg.per_judge_concurrency) for j in cfg.judges}
    stop = asyncio.Event()
    write_lock = asyncio.Lock()
    files = {t: (raw_dir / f"{t}.jsonl").open("a") for t in cfg.tasks}
    policy = RetryPolicy()

    async def handle(client: httpx.AsyncClient, job: Job) -> None:
        if stop.is_set():
            return
        spec = get_task(job.task).spec
        jc = job.judge.config
        body = job.judge.build_request(spec, job.state)
        rhash = request_hash(job.judge.endpoint, body, job.repeat)
        record = JudgmentRecord(
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            task=job.task,
            spec_version=spec.version,
            spec_hash=spec.spec_hash(),
            item_id=job.item.item_id,
            variant=job.variant_name,
            repeat=job.repeat,
            judge_id=jc.id,
            judge_kind=jc.kind,
            model_requested=jc.model,
            request_hash=rhash,
            gold=job.gold,
            approx_tokens=int(job.item.meta.get("approx_tokens", 0)) or None,
        )
        cached = cache.get(rhash)
        if cached is not None:
            resp_body, latency = cached.body, cached.latency_s
            record.cached = True
            summary.cached += 1
        else:
            async with global_sem, judge_sems[jc.id]:
                if stop.is_set():
                    return
                estimate = job.judge.estimate_cost(body)
                try:
                    budget.reserve(estimate)
                except BudgetExceeded as exc:
                    if not stop.is_set():
                        print(f"[budget] {exc} -- stopping", file=sys.stderr)
                    summary.budget_stopped = True
                    stop.set()
                    return
                try:
                    result = await post_json(client, job.judge.endpoint, body, headers, policy)
                except ProviderError as exc:
                    budget.release(estimate)
                    if exc.status == 402 or (exc.status == 403 and "limit" in str(exc).lower()):
                        # credits or key limit exhausted: stop cleanly instead of failing on
                        # every remaining job
                        print(f"[credits] {exc} -- stopping", file=sys.stderr)
                        summary.budget_stopped = True
                        stop.set()
                    record.error = str(exc)
                    record.raw_response = exc.body if isinstance(exc.body, dict) else None
                    summary.errors += 1
                    await _write(record)
                    return
                resp_body, latency = result.body, result.latency_s
                usage = resp_body.get("usage")
                actual: Any = (
                    cast(dict[str, Any], usage).get("cost") if isinstance(usage, dict) else None
                )
                cost = budget.settle(
                    estimate,
                    float(actual) if isinstance(actual, int | float) else None,
                    f"{jc.id}:{job.task}",
                )
                summary.paid += 1
                summary.spent_usd += cost
                summary.by_judge_cost[jc.id] += cost
        record.latency_s = latency
        record.raw_response = resp_body
        try:
            parsed = job.judge.parse(spec, resp_body)
        except ParseError as exc:
            record.error = f"parse error: {exc}"
            summary.errors += 1
            await _write(record)
            return
        if not record.cached:
            cache.put(rhash, resp_body, latency, record.timestamp)
        record.answers = parsed.answers
        record.model_served = parsed.model_served
        record.provider = parsed.provider
        record.cost_usd = parsed.cost_usd
        record.input_tokens = parsed.input_tokens
        record.output_tokens = parsed.output_tokens
        await _write(record)

    async def _write(record: JudgmentRecord) -> None:
        async with write_lock:
            fh = files[record.task]
            fh.write(record.model_dump_json() + "\n")
            fh.flush()

    limits = httpx.Limits(max_connections=cfg.concurrency * 2)
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_s, limits=limits) as client:
            total = len(pending)
            done_count = 0
            for chunk_start in range(0, total, 200):
                chunk = pending[chunk_start : chunk_start + 200]
                await asyncio.gather(*(handle(client, j) for j in chunk))
                done_count += len(chunk)
                print(
                    f"[{run_id}] {done_count}/{total} jobs | paid {summary.paid} "
                    f"cached {summary.cached} errors {summary.errors} | "
                    f"run ${summary.spent_usd:.4f} | ledger ${budget.spent_usd:.4f} "
                    f"of ${budget.cap_usd:.2f}",
                    file=sys.stderr,
                )
                if stop.is_set():
                    break
    finally:
        for fh in files.values():
            fh.close()
        cache.close()
    return summary
