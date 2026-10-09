"""Matplotlib charts (PNG). Decision models and generative judges are told apart by
both color and marker shape, so identity never rests on color alone."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from judgment_calls.report.analysis import PositionBias, ViewResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
FAMILY_COLOR = {"decision": "#2a78d6", "generative": "#eb6834"}
FAMILY_MARKER = {"decision": "o", "generative": "s"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)


def _usd(x: float, _pos: int | None = None) -> str:
    return f"${x:.2f}"


def _family(kind: str) -> str:
    return kind if kind in FAMILY_COLOR else "decision"


def accuracy_vs_cost(
    panels: list[tuple[str, str, list[tuple[ViewResult, float]]]], path: Path
) -> None:
    """panels: (title, metric key, [(result, cost per 1k judgments)])."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    axes_list: list[Axes] = list(axes) if len(panels) > 1 else [axes]
    for ax, (title, metric, rows) in zip(axes_list, panels, strict=True):
        placed: list[tuple[float, float]] = []
        span = max((r.metrics[metric].value for r, _ in rows), default=1.0) - min(
            (r.metrics[metric].value for r, _ in rows), default=0.0
        )
        for res, cost in sorted(rows, key=lambda rc: rc[1]):
            est = res.metrics[metric]
            fam = _family(res.view.kind)
            ax.errorbar(
                cost,
                est.value,
                yerr=[[est.value - est.lo], [est.hi - est.value]],
                fmt=FAMILY_MARKER[fam],
                color=FAMILY_COLOR[fam],
                ms=7,
                lw=1.5,
                capsize=3,
                mec=SURFACE,
                mew=1.5,
            )
            crowded = any(
                abs(math.log10(cost) - math.log10(px)) < 0.25 and abs(est.value - py) < 0.08 * span
                for px, py in placed
            )
            placed.append((cost, est.value))
            ax.annotate(
                res.view.label,
                (cost, est.value),
                xytext=(6, -11 if crowded else 4),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_2,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(_usd))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left")
        ax.set_xlabel("USD per 1,000 judgments (log scale)")
        ax.set_ylabel(metric.replace("_", " "))
    handles = [
        Line2D(
            [],
            [],
            marker=FAMILY_MARKER[f],
            color=FAMILY_COLOR[f],
            ls="",
            ms=7,
            label=f"{f} {'model' if f == 'decision' else 'judge'}",
        )
        for f in ("decision", "generative")
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle("Quality vs. cost (95% bootstrap CIs)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(results: list[ViewResult], title: str, path: Path) -> None:
    rs = [r for r in results if r.calibration is not None]
    cols = min(4, len(rs))
    rows = (len(rs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(rs):
            ax.axis("off")
            continue
        r = rs[k]
        cal = r.calibration
        assert cal is not None
        fam = _family(r.view.kind)
        ax.plot([0, 1], [0, 1], color=INK_2, lw=0.8, ls="--")
        xs = [b.mean_confidence for b in cal.bins]
        ys = [b.accuracy for b in cal.bins]
        ax.plot(
            xs,
            ys,
            color=FAMILY_COLOR[fam],
            lw=2,
            marker=FAMILY_MARKER[fam],
            ms=5,
            mec=SURFACE,
            mew=1,
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{r.view.label}\nECE {cal.ece.value:.3f}, n={r.n}", fontsize=8.5, loc="left")
        ax.set_xlabel("mean confidence (equal-mass bins)", fontsize=7.5)
        ax.set_ylabel("accuracy", fontsize=7.5)
        ax.set_aspect("equal")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_length(panels: list[tuple[str, str, list[ViewResult]]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    axes_list: list[Axes] = list(axes) if len(panels) > 1 else [axes]
    for ax, (title, ylabel, results) in zip(axes_list, panels, strict=True):
        for k, r in enumerate(results):
            if not r.by_length:
                continue
            fam = _family(r.view.kind)
            xs = list(range(len(r.by_length)))
            ys = [row.metric.value for row in r.by_length]
            ax.plot(
                xs,
                ys,
                color=SERIES[k % len(SERIES)],
                marker=FAMILY_MARKER[fam],
                lw=2,
                ms=6,
                mec=SURFACE,
                mew=1,
                label=r.view.label,
            )
            ax.set_xticks(xs, [f"{row.bucket}\n(n={row.n})" for row in r.by_length])
        ax.set_title(title, loc="left")
        ax.set_xlabel("input length bucket (approx. tokens)")
        ax.set_ylabel(ylabel)
    axes_list[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        "Quality by input length (point estimates; per-bucket CIs in REPORT.md)",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def position_bias_chart(rows: list[tuple[str, PositionBias]], path: Path) -> None:
    labels = [name for name, _ in rows]
    segs = [
        ("consistent", [pb.consistent for _, pb in rows], SERIES[0]),
        ("flips toward first-shown", [pb.favors_first for _, pb in rows], SERIES[1]),
        ("flips toward second-shown", [pb.favors_second for _, pb in rows], SERIES[2]),
        ("other flip (tie involved)", [pb.other_inconsistent for _, pb in rows], "#a8a69f"),
    ]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.6))
    left = [0.0] * len(rows)
    for name, vals, color in segs:
        ax.barh(
            labels,
            vals,
            left=left,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            label=name,
            height=0.6,
        )
        for i, v in enumerate(vals):
            if v >= 0.06:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if color != "#a8a69f" else INK,
                )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of comparisons judged in both A/B orders")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Position bias: does the verdict survive swapping A and B?", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
