"""Calibration metrics: ECE with equal-mass bins, Brier score, reliability bins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def equal_mass_bins(
    confidence: Sequence[float] | NDArray[np.float64], n_bins: int
) -> list[NDArray[np.intp]]:
    """Index arrays of ~equal size, sorted by confidence (stable for ties)."""
    c = np.asarray(confidence, dtype=np.float64)
    if len(c) == 0:
        return []
    order = np.argsort(c, kind="stable")
    k = max(1, min(n_bins, len(c)))
    return [chunk for chunk in np.array_split(order, k) if len(chunk)]


def reliability_bins(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[ReliabilityBin]:
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    out: list[ReliabilityBin] = []
    for idx in equal_mass_bins(c, n_bins):
        out.append(
            ReliabilityBin(
                lo=float(c[idx].min()),
                hi=float(c[idx].max()),
                count=len(idx),
                mean_confidence=float(c[idx].mean()),
                accuracy=float(y[idx].mean()),
            )
        )
    return out


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Top-label ECE with equal-mass (adaptive) bins: sum_b |B|/n * |acc_b - conf_b|."""
    n = len(confidence)
    if n == 0:
        return float("nan")
    bins = reliability_bins(confidence, correct, n_bins)
    return float(sum(b.count / n * abs(b.accuracy - b.mean_confidence) for b in bins))


def brier_score(prob: Sequence[float], outcome: Sequence[bool]) -> float:
    """Mean squared error between a probability and a binary outcome."""
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def wrong_at_confidence(
    confidence: Sequence[float], correct: Sequence[bool], threshold: float = 0.9
) -> tuple[float, float, int]:
    """(share of all predictions that are wrong with confidence >= threshold,
    error rate among predictions with confidence >= threshold, count >= threshold)."""
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct, dtype=bool)
    if len(c) == 0:
        return float("nan"), float("nan"), 0
    high = c >= threshold
    n_high = int(high.sum())
    wrong_high = int((high & ~y).sum())
    return wrong_high / len(c), (wrong_high / n_high if n_high else float("nan")), n_high
