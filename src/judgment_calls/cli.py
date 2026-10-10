"""Command line interface: ``judgment-calls prepare|run|report|list-models|list-judges``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from judgment_calls.judges import JUDGES, get_judges
from judgment_calls.tasks import TASKS

DEFAULT_BUDGET_USD = 8.0


def _cmd_prepare(args: argparse.Namespace) -> int:
    from judgment_calls.datasets.prepare import prepare_all

    manifest = prepare_all(
        args.tasks, Path(args.raw_dir), Path(args.samples_dir), args.n, args.seed
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from judgment_calls.runner import RunConfig, run

    cfg = RunConfig(
        tasks=args.tasks,
        judges=get_judges(args.judges),
        samples_dir=Path(args.samples_dir),
        results_dir=Path(args.results_dir),
        cache_path=Path(args.cache),
        budget_cap_usd=args.budget,
        limit=args.limit,
        repeats=args.repeats,
        repeat_limit=args.repeat_limit,
        concurrency=args.concurrency,
        per_judge_concurrency=args.per_judge_concurrency,
        dry_run=args.dry_run,
        subset_dir=Path(args.subset_dir) if args.subset_dir else None,
    )
    summary = asyncio.run(run(cfg))
    out = {
        "run_id": summary.run_id,
        "planned": summary.planned,
        "already_done": summary.skipped_done,
        "cached": summary.cached,
        "paid_calls": summary.paid,
        "errors": summary.errors,
        "spent_usd": round(summary.spent_usd, 6),
        "budget_stopped": summary.budget_stopped,
        "by_judge_cost": {k: round(v, 6) for k, v in summary.by_judge_cost.items()},
    }
    if args.dry_run:
        out = {
            "planned": summary.planned,
            "already_done": summary.skipped_done,
            "estimated_max_usd": round(summary.estimated_usd, 4),
        }
    print(json.dumps(out, indent=2))
    return 1 if summary.budget_stopped else 0


def _cmd_subset(args: argparse.Namespace) -> int:
    from judgment_calls.datasets.prepare import write_reference_subset

    sizes = {"faithfulness": args.faithfulness, "summeval": args.summeval}
    out = write_reference_subset(Path(args.samples_dir), Path(args.out_dir), sizes, args.seed)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from judgment_calls.report.build import build_report

    path = build_report(Path(args.results_dir), Path(args.samples_dir))
    print(f"wrote {path}")
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    headers: dict[str, str] = {}
    if key := os.environ.get("OPENROUTER_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    resp = httpx.get(
        "https://openrouter.ai/api/v1/models",
        params={"output_modalities": "decisions"},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    for m in resp.json()["data"]:
        price = float(m.get("pricing", {}).get("prompt") or 0) * 1e6
        print(f"{m['id']:45s} ctx={m.get('context_length')!s:>8}  ${price:.3f}/M input")
    return 0


def _cmd_list_judges(args: argparse.Namespace) -> int:
    for j in JUDGES:
        print(f"{j.id:22s} {j.kind:10s} {j.tier:9s} {j.model:40s} {j.notes}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judgment-calls", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    task_names = sorted(TASKS)

    pp = sub.add_parser("prepare", help="download datasets and build seeded samples")
    pp.add_argument("--tasks", nargs="+", default=task_names, choices=task_names)
    pp.add_argument("--n", type=int, default=200)
    pp.add_argument("--seed", type=int, default=20261009)
    pp.add_argument("--raw-dir", default="data/raw")
    pp.add_argument("--samples-dir", default="data/samples")
    pp.set_defaults(fn=_cmd_prepare)

    pr = sub.add_parser("run", help="run judges (cached, budget-capped)")
    pr.add_argument("--tasks", nargs="+", default=task_names, choices=task_names)
    pr.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="judge ids, or 'primary' / 'all-decision' / 'all-generative' (default: primary)",
    )
    pr.add_argument("--limit", type=int, default=None, help="items per task (hash order)")
    pr.add_argument("--repeats", type=int, default=1)
    pr.add_argument(
        "--repeat-limit",
        type=int,
        default=None,
        help="only the first N items get the extra repeats",
    )
    pr.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help="hard cap in USD over the whole spend ledger",
    )
    pr.add_argument("--concurrency", type=int, default=16)
    pr.add_argument("--per-judge-concurrency", type=int, default=6)
    pr.add_argument("--samples-dir", default="data/samples")
    pr.add_argument("--results-dir", default="results")
    pr.add_argument("--cache", default=".cache/responses.sqlite")
    pr.add_argument("--dry-run", action="store_true", help="estimate cost, call nothing")
    pr.add_argument(
        "--subset-dir",
        default=None,
        help="only run items listed in <dir>/<task>.txt (e.g. data/samples/reference_subset)",
    )
    pr.set_defaults(fn=_cmd_run)

    ps = sub.add_parser("subset", help="write a fixed stratified item subset (id lists)")
    ps.add_argument("--faithfulness", type=int, default=60)
    ps.add_argument("--summeval", type=int, default=40)
    ps.add_argument("--seed", type=int, default=20261010)
    ps.add_argument("--samples-dir", default="data/samples")
    ps.add_argument("--out-dir", default="data/samples/reference_subset")
    ps.set_defaults(fn=_cmd_subset)

    prep = sub.add_parser("report", help="compute metrics, write results/REPORT.md + charts")
    prep.add_argument("--results-dir", default="results")
    prep.add_argument("--samples-dir", default="data/samples")
    prep.set_defaults(fn=_cmd_report)

    pl = sub.add_parser("list-models", help="list OpenRouter decision models")
    pl.set_defaults(fn=_cmd_list_models)
    pj = sub.add_parser("list-judges", help="list configured judges")
    pj.set_defaults(fn=_cmd_list_judges)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
