"""Assemble results/REPORT.md, results/summary.json and charts from the raw logs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from judgment_calls.datasets.sampling import bucket_label, length_bucket
from judgment_calls.judges import JUDGES
from judgment_calls.metrics.uncertainty import Estimate
from judgment_calls.models import JudgmentRecord
from judgment_calls.report import charts
from judgment_calls.report.analysis import (
    MIN_N,
    SUMMEVAL_DIMS,
    Comparison,
    CostLatency,
    PositionBias,
    View,
    ViewResult,
    analyse_noul,
    analyse_pairwise,
    analyse_score,
    compare_noul,
    compare_pairwise,
    compare_score,
    cost_latency,
    holm,
    repeat_stats,
    views_for,
)
from judgment_calls.report.load import TaskResults, load_task
from judgment_calls.tasks import TASKS

ORDER = {j.id: k for k, j in enumerate(JUDGES)}
HEADLINE = {
    "faithfulness": "balanced_accuracy",
    "summeval": "mean_spearman",
    "pairwise": "accuracy",
}


def _sort_views(views: list[View]) -> list[View]:
    return sorted(views, key=lambda v: (ORDER.get(v.judge_id, 99), v.signal))


def _f(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.{d}f}"


def _p(x: float) -> str:
    if math.isnan(x):
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.0%}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _est_json(e: Estimate) -> dict[str, float]:
    return {"value": e.value, "lo": e.lo, "hi": e.hi, "n": e.n}


def _kind(v: View) -> str:
    return "decision" if v.kind == "decision" else "generative"


def _name(v: View) -> str:
    return f"{v.label}" + (" ¹" if _partial.get(v.judge_id) else "")


_partial: dict[str, bool] = {}


def build_report(results_dir: Path, samples_dir: Path) -> Path:
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    trs: dict[str, TaskResults] = {t: load_task(t, results_dir, samples_dir) for t in TASKS}
    all_records: list[JudgmentRecord] = [r for tr in trs.values() for r in tr.records.values()]
    manifest: dict[str, Any] = json.loads((samples_dir / "manifest.json").read_text())

    # ----- per-task analysis ------------------------------------------------
    results: dict[str, dict[View, ViewResult]] = {t: {} for t in TASKS}
    pbias: dict[View, PositionBias] = {}
    smoke_only: dict[str, list[str]] = {t: [] for t in TASKS}
    analysers: dict[str, Callable[[TaskResults, View], ViewResult]] = {
        "faithfulness": analyse_noul,
        "summeval": analyse_score,
    }
    for task, tr in trs.items():
        for v in _sort_views(views_for(tr)):
            if task == "pairwise":
                res, pb = analyse_pairwise(tr, v)
            else:
                res = analysers[task](tr, v)
                pb = None
            if res.n < MIN_N:
                smoke_only[task].append(f"{v.label} (n={res.n})")
                continue
            results[task][v] = res
            if pb is not None:
                pbias[v] = pb
            if res.n < len(tr.items):
                _partial[v.judge_id] = True

    # ----- cost & latency -----------------------------------------------------
    judge_ids = sorted({r.judge_id for r in all_records}, key=lambda j: ORDER.get(j, 99))
    cost_all = {j: cost_latency(all_records, j) for j in judge_ids}
    cost_task: dict[str, dict[str, CostLatency]] = {
        t: {j: cost_latency(list(tr.records.values()), j) for j in judge_ids}
        for t, tr in trs.items()
    }

    # ----- paired comparisons -------------------------------------------------
    comparisons: list[Comparison] = []
    comparers: dict[str, Callable[[TaskResults, View, View], list[Comparison]]] = {
        "faithfulness": compare_noul,
        "summeval": compare_score,
        "pairwise": compare_pairwise,
    }
    for task, tr in trs.items():
        dec = [v for v in results[task] if _kind(v) == "decision"]
        gen = [v for v in results[task] if _kind(v) == "generative" and v.signal == "native"]
        family: list[Comparison] = []
        for a in dec:
            for b in gen:
                family.extend(comparers[task](tr, a, b))
        for metric in {c.metric for c in family}:
            fam = [c for c in family if c.metric == metric]
            for c, p in zip(fam, holm([c.diff.p_value for c in fam]), strict=True):
                c.p_holm = p
        comparisons.extend(family)

    # ----- repeats ------------------------------------------------------------
    rep_rows: list[list[str]] = []
    rep_json: list[dict[str, Any]] = []
    for task, tr in trs.items():
        for j in judge_ids:
            st = repeat_stats(tr, j)
            if st is None:
                continue
            rep_rows.append(
                [
                    j,
                    task,
                    str(st.n_groups),
                    str(st.n_answers),
                    _pct(st.flip_rate),
                    _f(st.mean_std, 4),
                ]
            )
            rep_json.append(st.__dict__)

    # ----- charts ---------------------------------------------------------------
    panels = [
        (
            "Faithfulness (RAGTruth)",
            "balanced_accuracy",
            [
                (r, cost_task["faithfulness"][v.judge_id].cost_per_1k)
                for v, r in results["faithfulness"].items()
                if v.signal == "native"
            ],
        ),
        (
            "Rubric scoring (SummEval)",
            "mean_spearman",
            [
                (r, cost_task["summeval"][v.judge_id].cost_per_1k)
                for v, r in results["summeval"].items()
                if v.signal == "native"
            ],
        ),
        (
            "Pairwise preference (MT-Bench)",
            "accuracy",
            [
                (r, cost_task["pairwise"][v.judge_id].cost_per_1k)
                for v, r in results["pairwise"].items()
                if v.signal == "native"
            ],
        ),
    ]
    charts.accuracy_vs_cost(panels, charts_dir / "accuracy_vs_cost.png")
    charts.reliability(
        list(results["faithfulness"].values()),
        "Reliability: faithfulness (top-label confidence)",
        charts_dir / "reliability_faithfulness.png",
    )
    charts.reliability(
        list(results["pairwise"].values()),
        "Reliability: pairwise preference (top-label confidence, both orders)",
        charts_dir / "reliability_pairwise.png",
    )
    full = {t: [r for v, r in results[t].items() if not _partial.get(v.judge_id)] for t in TASKS}
    charts.by_length(
        [
            ("Faithfulness", "balanced accuracy", full["faithfulness"]),
            ("SummEval", "mean Spearman ρ", full["summeval"]),
            ("Pairwise", "accuracy (both orders)", full["pairwise"]),
        ],
        charts_dir / "accuracy_by_length.png",
    )
    charts.position_bias_chart(
        [(v.label, pb) for v, pb in pbias.items()], charts_dir / "position_bias.png"
    )

    # ----- markdown -------------------------------------------------------------
    spend = 0.0
    ledger = results_dir / "spend_ledger.jsonl"
    if ledger.exists():
        spend = sum(
            float(json.loads(x)["cost_usd"]) for x in ledger.read_text().splitlines() if x.strip()
        )
    md: list[str] = []
    md.append("# judgment-calls: results report\n")
    md.append(
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `judgment-calls report` "
        f"from `results/raw/*.jsonl`. Total API spend recorded in the ledger: **${spend:.4f}**. "
        "All intervals are 95% percentile-bootstrap CIs over items (2,000 resamples). "
        "Headline metrics use repeat 0 of every item.\n"
    )
    md.append(
        "¹ = judge covered only part of the sample (the run hit the API key's credit "
        "limit); its numbers are on the items it did judge and are **not** directly "
        "comparable to full-sample judges except in the paired comparisons, which always "
        "use common items.\n"
    )

    # coverage
    md.append("## Coverage\n")
    cov_rows: list[list[str]] = []
    for j in judge_ids:
        row = [j]
        for tr in trs.values():
            n_items = len({k[1] for k in tr.records if k[0] == j and k[3] == 0})
            row.append(f"{n_items}/{len(tr.items)}")
        cov_rows.append(row)
    md.append(_table(["judge", *TASKS.keys()], cov_rows) + "\n")
    for t in TASKS:
        if smoke_only[t]:
            md.append(f"- {t}: smoke-test only, excluded from analysis: {', '.join(smoke_only[t])}")
    md.append("")

    # headline
    md.append("## Headline\n")
    hrows: list[list[str]] = []
    labels = _sort_views(list({v for t in TASKS for v in results[t]}))
    for v in labels:
        if v.signal != "native":
            continue
        cells = [_name(v), _kind(v)]
        for t, key in [
            ("faithfulness", "balanced_accuracy"),
            ("faithfulness", "auroc"),
            ("summeval", "mean_spearman"),
            ("pairwise", "accuracy"),
        ]:
            r = results[t].get(v)
            cells.append(r.metrics[key].fmt() if r else "—")
        cl = cost_all[v.judge_id]
        cells += [f"${cl.cost_per_1k:.3f}", f"{cl.p50_s:.2f}s"]
        hrows.append(cells)
    md.append(
        _table(
            [
                "judge",
                "family",
                "faithfulness bal. acc.",
                "faithfulness AUROC",
                "SummEval mean ρ",
                "pairwise acc. (A/B/tie)",
                "$/1k judgments",
                "p50 latency",
            ],
            hrows,
        )
        + "\n"
    )
    md.append("![quality vs cost](charts/accuracy_vs_cost.png)\n")

    # faithfulness
    md.append("## Task 1: faithfulness / hallucination detection (RAGTruth, `noul`)\n")
    fm = manifest.get("faithfulness", {}).get("sample", {})
    md.append(
        f"Sample: {fm.get('n')} responses, labels {fm.get('labels')}, length buckets "
        f"{fm.get('length_buckets')}. Positive class = hallucinated.\n"
    )
    rows: list[list[str]] = []
    for v, r in results["faithfulness"].items():
        c = r.calibration
        assert c is not None
        rows.append(
            [
                _name(v),
                str(r.n),
                r.metrics["accuracy"].fmt(),
                r.metrics["balanced_accuracy"].fmt(),
                r.metrics["macro_f1"].fmt(),
                r.metrics["kappa"].fmt(),
                r.metrics["auroc"].fmt(),
                _pct(r.extra["flagged_rate"]),
                _pct(r.extra["recall_halluc"]),
                _pct(r.extra["precision_halluc"]),
            ]
        )
    md.append(
        _table(
            [
                "judge",
                "n",
                "accuracy",
                "balanced acc.",
                "macro-F1",
                "Cohen κ",
                "AUROC",
                "flagged",
                "recall",
                "precision",
            ],
            rows,
        )
        + "\n"
    )
    md.append("Accuracy by RAGTruth sub-task:\n")
    groups = sorted({g for r in results["faithfulness"].values() for g in r.by_group})
    md.append(
        _table(
            ["judge", *groups],
            [
                [_name(v), *[r.by_group[g].fmt(2) if g in r.by_group else "—" for g in groups]]
                for v, r in results["faithfulness"].items()
            ],
        )
        + "\n"
    )

    # calibration
    md.append("## Calibration\n")
    md.append(
        "Top-label confidence vs. correctness. Decision models: the model's probability "
        "for its answer. Generative judges: verbalized confidence (0–100 → 0–1), or the "
        "token probability of the answer for `(logprob)` rows. ECE uses 10 equal-mass bins. "
        "Brier: faithfulness uses P(hallucinated) vs. the label; pairwise uses top-label "
        "confidence vs. correctness. *wrong@≥0.9* = share of all judgments that are wrong "
        "while claiming ≥90% confidence. Note: ECE is biased upward under resampling, so "
        "its bootstrap interval can sit mostly above the point estimate; use it as a rough "
        "guide and prefer the paired Brier/ECE comparisons below.\n"
    )
    rows = []
    for t in ("faithfulness", "pairwise"):
        for v, r in results[t].items():
            c = r.calibration
            assert c is not None
            rows.append(
                [
                    t,
                    _name(v),
                    str(r.n),
                    _f(c.mean_confidence),
                    c.ece.fmt(),
                    c.brier.fmt(),
                    _pct(c.wrong_at_90),
                    f"{_pct(c.error_rate_at_90)} of {c.n_at_90}",
                ]
            )
    md.append(
        _table(
            [
                "task",
                "judge",
                "n",
                "mean conf.",
                "ECE",
                "Brier",
                "wrong@≥0.9",
                "error rate when ≥0.9",
            ],
            rows,
        )
        + "\n"
    )
    md.append("![reliability faithfulness](charts/reliability_faithfulness.png)\n")
    md.append("![reliability pairwise](charts/reliability_pairwise.png)\n")

    # summeval
    md.append("## Task 2: rubric scoring (SummEval, `score`)\n")
    sm = manifest.get("summeval", {}).get("sample", {})
    md.append(
        f"Sample: {sm.get('n')} summaries (stratified by the rounded mean expert rating: "
        f"{sm.get('labels')} on the 0–4 index scale). One call per summary answers all four "
        "dimensions. Decision models: expected score Σ i·p(i); generative judges: the integer "
        "they output (or the logprob expectation). Correlations are pooled over summaries "
        "(summary-level), not averaged per document.\n"
    )
    rows = []
    for v, r in results["summeval"].items():
        rows.append(
            [
                _name(v),
                str(r.n),
                r.metrics["mean_spearman"].fmt(),
                r.metrics["mean_kendall"].fmt(),
                r.metrics["mean_qwk"].fmt(),
                r.metrics["mean_mae"].fmt(),
                _f(r.extra["bias"], 2),
            ]
        )
    md.append(
        _table(
            [
                "judge",
                "n",
                "mean Spearman ρ",
                "mean Kendall τ",
                "mean QWK",
                "mean MAE",
                "mean bias (judge − human)",
            ],
            rows,
        )
        + "\n"
    )
    md.append("Spearman ρ per dimension:\n")
    md.append(
        _table(
            ["judge", *SUMMEVAL_DIMS],
            [
                [_name(v), *[r.metrics[f"spearman_{d}"].fmt(2) for d in SUMMEVAL_DIMS]]
                for v, r in results["summeval"].items()
            ],
        )
        + "\n"
    )

    # pairwise
    pw = manifest.get("pairwise", {})
    md.append("## Task 3: pairwise preference (MT-Bench human judgments, `choice`)\n")
    md.append(
        f"Sample: {pw.get('sample', {}).get('n')} comparisons (labels "
        f"{pw.get('sample', {}).get('labels')}), each judged in both A/B orders. Human–human "
        f"agreement on multiply-annotated comparisons in the full dataset: "
        f"**{_pct(pw.get('human_pair_agreement_with_ties', float('nan')))} with ties** "
        f"({pw.get('human_pairs_with_ties')} vote pairs) and "
        f"**{_pct(pw.get('human_pair_agreement_without_ties', float('nan')))} without ties** "
        f"({pw.get('human_pairs_without_ties')} pairs), a practical ceiling.\n"
    )
    rows = []
    for v, r in results["pairwise"].items():
        pb = pbias[v]
        rows.append(
            [
                _name(v),
                str(r.n),
                r.metrics["accuracy"].fmt(),
                r.metrics["agreement_no_tie"].fmt(),
                r.metrics["macro_f1"].fmt(),
                r.metrics["kappa"].fmt(),
                _pct(r.extra["tie_rate_pred"]),
                r.metrics["consistency"].fmt(),
                _pct(pb.favors_first),
                _pct(pb.favors_second),
                _f(pb.first_position_rate, 2),
            ]
        )
    md.append(
        _table(
            [
                "judge",
                "n",
                "accuracy (3-way)",
                "agreement w/o ties",
                "macro-F1",
                "κ",
                "predicts tie",
                "A/B-swap consistency",
                "flips → first",
                "flips → second",
                "P(first | non-tie)",
            ],
            rows,
        )
        + "\n"
    )
    md.append("![position bias](charts/position_bias.png)\n")
    md.append("Accuracy by turn:\n")
    md.append(
        _table(
            ["judge", "turn 1", "turn 2"],
            [
                [
                    _name(v),
                    *[
                        r.by_group[g].fmt(2) if g in r.by_group else "—"
                        for g in ("turn 1", "turn 2")
                    ],
                ]
                for v, r in results["pairwise"].items()
            ],
        )
        + "\n"
    )

    # length
    md.append("## Input length\n")
    md.append(
        "Approximate tokens = characters / 4 over all state fields. Samples were "
        "stratified to over-represent long inputs. Per-bucket n is small; read the CIs.\n"
    )
    for t, label in [
        ("faithfulness", "balanced accuracy"),
        ("summeval", "mean Spearman ρ"),
        ("pairwise", "accuracy"),
    ]:
        buckets = sorted(
            {row.bucket for r in results[t].values() for row in r.by_length},
            key=lambda b: int(b.split("-")[0].lstrip(">=")),
        )
        md.append(f"**{t}** ({label}):\n")
        md.append(
            _table(
                ["judge", *buckets],
                [
                    [
                        _name(v),
                        *[
                            next(
                                (
                                    f"{row.metric.fmt(2)} (n={row.n})"
                                    for row in r.by_length
                                    if row.bucket == b
                                ),
                                "—",
                            )
                            for b in buckets
                        ],
                    ]
                    for v, r in results[t].items()
                ],
            )
            + "\n"
        )
    tie_share: dict[str, list[bool]] = {}
    for it in trs["pairwise"].items.values():
        b = bucket_label(length_bucket(int(it.meta.get("approx_tokens", 0))))
        tie_share.setdefault(b, []).append(it.gold["preference"] == "tie")
    md.append(
        "Caveat for pairwise: the buckets differ in label mix, and judges rarely predict *tie*, "
        "so buckets with fewer human ties look easier. Human-tie share per bucket: "
        + ", ".join(
            f"{b}: {sum(v) / len(v):.0%} (n={len(v)})"
            for b, v in sorted(
                tie_share.items(), key=lambda kv: int(kv[0].split("-")[0].lstrip(">="))
            )
        )
        + ". Movements shared by every judge are therefore item-mix effects, not length effects.\n"
    )
    md.append("![accuracy by length](charts/accuracy_by_length.png)\n")

    # repeats
    md.append("## Consistency across repeats\n")
    md.append(
        "Three identical calls (temperature 0 for generative judges) on the first 10 "
        "hash-ordered items of each task (pairwise: both orders). *flip rate* = share of "
        "answers whose label is not identical across the 3 calls; *mean std* = standard "
        "deviation across calls of the probability of the first answer's label (decision "
        "models), of the verbalized confidence (generative), or of the expected score "
        "(SummEval).\n"
    )
    md.append(
        _table(["judge", "task", "groups", "answers", "flip rate", "mean std"], rep_rows) + "\n"
    )

    # cost
    md.append("## Cost and latency\n")
    md.append(
        "Cost is the sum of `usage.cost` reported by OpenRouter per call. A *judgment* is "
        "one API call (SummEval: one call answers 4 rubric questions; pairwise: each order "
        "is one call). Latency is wall-clock per successful call from this machine "
        "(Europe), with up to 32 concurrent requests.\n"
    )
    rows = []
    for j in judge_ids:
        cl = cost_all[j]
        per_task = [
            f"${cost_task[t][j].cost_per_1k:.3f}" if cost_task[t][j].n_calls else "—" for t in TASKS
        ]
        rows.append(
            [
                j,
                str(cl.n_calls),
                f"${cl.total_cost:.4f}",
                f"${cl.cost_per_1k:.3f}",
                *per_task,
                _f(cl.in_tokens_mean, 0),
                f"{cl.p50_s:.2f}s",
                f"{cl.p95_s:.2f}s",
            ]
        )
    md.append(
        _table(
            [
                "judge",
                "calls",
                "total $",
                "$/1k (all)",
                *[f"$/1k {t}" for t in TASKS],
                "mean input tokens",
                "p50",
                "p95",
            ],
            rows,
        )
        + "\n"
    )

    # comparisons
    md.append("## Paired comparisons: decision model vs. generative judge\n")
    md.append(
        "Difference = decision − generative on the items both judged, with a paired "
        "bootstrap 95% CI and two-sided bootstrap p; Holm-adjusted across all "
        "decision×generative pairs of a task and metric. McNemar (exact) on per-item "
        "correctness is shown for binary/one-order correctness (pairwise: original order "
        "only). A difference is called significant only if Holm p < 0.05.\n"
    )
    rows = []
    for c in comparisons:
        sig = "**yes**" if (c.p_holm is not None and c.p_holm < 0.05) else "no"
        mc = f"{c.a_only}/{c.b_only}, p={_p(c.mcnemar_p)}" if c.mcnemar_p is not None else "—"
        rows.append(
            [
                c.task,
                c.metric,
                c.a,
                c.b,
                str(c.n),
                f"{c.diff.diff:+.3f} [{c.diff.lo:+.3f}, {c.diff.hi:+.3f}]",
                _p(c.diff.p_value),
                _p(c.p_holm if c.p_holm is not None else float("nan")),
                sig,
                mc,
            ]
        )
    md.append(
        _table(
            [
                "task",
                "metric",
                "decision",
                "generative",
                "n",
                "Δ [95% CI]",
                "p",
                "p (Holm)",
                "significant",
                "McNemar dec-only/gen-only",
            ],
            rows,
        )
        + "\n"
    )

    report = results_dir / "REPORT.md"
    report.write_text("\n".join(md) + "\n")

    summary: dict[str, Any] = {
        "spend_usd": spend,
        "results": {
            t: {
                v.label: {
                    "kind": _kind(v),
                    "n": r.n,
                    "partial": bool(_partial.get(v.judge_id)),
                    "metrics": {k: _est_json(e) for k, e in r.metrics.items()},
                    "extra": r.extra,
                    "calibration": None
                    if r.calibration is None
                    else {
                        "ece": _est_json(r.calibration.ece),
                        "brier": _est_json(r.calibration.brier),
                        "wrong_at_90": r.calibration.wrong_at_90,
                    },
                }
                for v, r in results[t].items()
            }
            for t in TASKS
        },
        "position_bias": {v.label: pb.__dict__ for v, pb in pbias.items()},
        "cost": {j: c.__dict__ for j, c in cost_all.items()},
        "repeats": rep_json,
        "comparisons": [
            {
                "task": c.task,
                "metric": c.metric,
                "a": c.a,
                "b": c.b,
                "n": c.n,
                "diff": c.diff.diff,
                "lo": c.diff.lo,
                "hi": c.diff.hi,
                "p": c.diff.p_value,
                "p_holm": c.p_holm,
                "mcnemar_p": c.mcnemar_p,
            }
            for c in comparisons
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return report
