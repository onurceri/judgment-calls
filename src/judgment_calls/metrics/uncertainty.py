"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Metric = Callable[[NDArray[np.int64]], float]
"""A metric evaluated on a vector of resampled row indices."""


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n: int

    def fmt(self, digits: int = 3) -> str:
        if np.isnan(self.value):
            return "n/a"
        return f"{self.value:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def bootstrap_ci(
    metric: Metric, n: int, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Percentile bootstrap over items (rows resampled with replacement)."""
    idx = np.arange(n, dtype=np.int64)
    point = metric(idx)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    samples = np.array([metric(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=np.float64)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return Estimate(point, float("nan"), float("nan"), n)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(point, float(lo), float(hi), n)


@dataclass(frozen=True)
class PairedDiff:
    diff: float
    lo: float
    hi: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        return not (self.lo <= 0.0 <= self.hi)


def paired_bootstrap(
    metric_a: Metric,
    metric_b: Metric,
    n: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiff:
    """Bootstrap the difference metric_a - metric_b on the *same* resampled items.

    p_value: two-sided, from the share of bootstrap differences on the other side of 0.
    """
    idx = np.arange(n, dtype=np.int64)
    diff = metric_a(idx) - metric_b(idx)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[b] = metric_a(s) - metric_b(s)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return PairedDiff(float(diff), float(lo), float(hi), min(1.0, p), n)


@dataclass(frozen=True)
class McNemar:
    a_only: int  # a correct, b wrong
    b_only: int  # b correct, a wrong
    p_value: float


def mcnemar_exact(correct_a: NDArray[np.bool_], correct_b: NDArray[np.bool_]) -> McNemar:
    """Exact (binomial) McNemar test on paired correctness."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    k = a_only + b_only
    p = 1.0 if k == 0 else float(stats.binomtest(min(a_only, b_only), k, 0.5).pvalue)
    return McNemar(a_only, b_only, p)
