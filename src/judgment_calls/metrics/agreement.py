"""Label-agreement metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray


def _arr(x: Sequence[object] | NDArray[np.generic]) -> NDArray[np.object_]:
    return np.asarray(x, dtype=object)


def _eq(a: NDArray[np.object_], b: object) -> NDArray[np.bool_]:
    return np.asarray(a == b, dtype=bool)


def _values(a: NDArray[np.object_]) -> list[object]:
    return cast(list[object], a.tolist())


def accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    t, p = _arr(y_true), _arr(y_pred)
    if len(t) == 0:
        return float("nan")
    return float(np.mean(_eq(t, p)))


def balanced_accuracy(
    y_true: Sequence[object], y_pred: Sequence[object], labels: Sequence[object] | None = None
) -> float:
    """Mean per-class recall over classes present in y_true."""
    t, p = _arr(y_true), _arr(y_pred)
    classes = list(labels) if labels is not None else sorted(set(_values(t)), key=str)
    recalls = [float(np.mean(_eq(p[_eq(t, c)], c))) for c in classes if np.any(_eq(t, c))]
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(
    y_true: Sequence[object], y_pred: Sequence[object], labels: Sequence[object] | None = None
) -> float:
    t, p = _arr(y_true), _arr(y_pred)
    classes = (
        list(labels) if labels is not None else sorted(set(_values(t)) | set(_values(p)), key=str)
    )
    f1s: list[float] = []
    for c in classes:
        pc, tc = _eq(p, c), _eq(t, c)
        tp = float(np.sum(pc & tc))
        fp = float(np.sum(pc & ~tc))
        fn = float(np.sum(~pc & tc))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def confusion_matrix(
    y_true: Sequence[object], y_pred: Sequence[object], labels: Sequence[object]
) -> NDArray[np.float64]:
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for t, p in zip(y_true, y_pred, strict=True):
        m[index[t], index[p]] += 1
    return m


def cohen_kappa(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[object],
    weights: str | None = None,
) -> float:
    """Cohen's kappa; weights=None (nominal), "linear" or "quadratic" (QWK).

    For weighted variants, `labels` must be in ordinal order.
    """
    m = confusion_matrix(y_true, y_pred, labels)
    n = m.sum()
    if n == 0:
        return float("nan")
    k = len(labels)
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    if weights is None:
        w = (i != j).astype(np.float64)
    elif weights == "linear":
        w = np.abs(i - j).astype(np.float64) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) ** 2).astype(np.float64) / max(k - 1, 1) ** 2
    else:
        raise ValueError(weights)
    expected = np.outer(m.sum(axis=1), m.sum(axis=0)) / n
    denom = float((w * expected).sum())
    if denom == 0:
        return 1.0 if float((w * m).sum()) == 0 else 0.0
    return float(1.0 - (w * m).sum() / denom)


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], n_levels: int) -> float:
    return cohen_kappa(list(y_true), list(y_pred), list(range(n_levels)), weights="quadratic")
