"""Threshold-free and rank-correlation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def auroc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count 1/2)."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)  # average ranks for ties
    u = float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.kendalltau(x, y).statistic)  # tau-b


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")
