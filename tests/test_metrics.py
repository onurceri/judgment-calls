from __future__ import annotations

import math

import numpy as np
import pytest

from judgment_calls.metrics.agreement import (
    accuracy,
    balanced_accuracy,
    cohen_kappa,
    macro_f1,
    quadratic_weighted_kappa,
)
from judgment_calls.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    wrong_at_confidence,
)
from judgment_calls.metrics.ranking import auroc, kendall_tau, mae, spearman
from judgment_calls.metrics.uncertainty import bootstrap_ci, mcnemar_exact, paired_bootstrap


def test_accuracy_and_balanced_accuracy() -> None:
    y = [1, 1, 1, 1, 0]
    p = [1, 1, 1, 1, 1]
    assert accuracy(y, p) == pytest.approx(0.8)
    assert balanced_accuracy(y, p) == pytest.approx(0.5)  # recall 1.0 and 0.0


def test_macro_f1() -> None:
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    # class a: tp=1 fp=0 fn=1 -> 2/3 ; class b: tp=2 fp=1 fn=0 -> 0.8
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 0.8) / 2)


def test_cohen_kappa_known_value() -> None:
    # 2x2 table (rows = truth, cols = prediction): [[20, 5], [10, 15]]
    y = [1] * 25 + [0] * 25
    p = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert accuracy(y, p) == pytest.approx(0.7)
    # pe = 0.5 * 0.6 + 0.5 * 0.4 = 0.5 -> kappa = (0.7 - 0.5) / (1 - 0.5)
    assert cohen_kappa(y, p, [0, 1]) == pytest.approx(0.4)


def test_kappa_perfect_and_chance() -> None:
    assert cohen_kappa([0, 1, 2], [0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert cohen_kappa([0, 1, 0, 1], [0, 0, 1, 1], [0, 1]) == pytest.approx(0.0)


def test_qwk_matches_reference_implementation() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 0, 4)
    # reference: direct formula 1 - sum(w O) / sum(w E)
    o = np.zeros((5, 5))
    for i, j in zip(a, b, strict=True):
        o[i, j] += 1
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = np.array([[(i - j) ** 2 / 16 for j in range(5)] for i in range(5)])
    ref = 1 - (w * o).sum() / (w * e).sum()
    assert quadratic_weighted_kappa(a.tolist(), b.tolist(), 5) == pytest.approx(ref)
    assert quadratic_weighted_kappa([0, 4], [0, 4], 5) == pytest.approx(1.0)


def test_auroc() -> None:
    assert auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert auroc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    assert auroc([True, False], [0.5, 0.5]) == pytest.approx(0.5)  # ties count 1/2
    # 3 pos, 2 neg; pairs where pos > neg: (0.8>0.3,0.8>0.6,0.4>0.3,0.7>0.3,0.7>0.6) = 5 of 6
    assert auroc([True, True, True, False, False], [0.8, 0.4, 0.7, 0.3, 0.6]) == pytest.approx(
        5 / 6
    )
    assert math.isnan(auroc([True, True], [0.1, 0.2]))


def test_rank_correlations_and_mae() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    assert spearman(x, [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert kendall_tau(x, [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert math.isnan(spearman(x, [1.0, 1.0, 1.0, 1.0]))
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_brier() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    assert brier_score([0.8], [False]) == pytest.approx(0.64)


def test_ece_equal_mass() -> None:
    # perfectly calibrated within each bin -> 0
    conf = [0.5] * 4 + [1.0] * 4
    correct = [True, False, True, False, True, True, True, True]
    assert expected_calibration_error(conf, correct, n_bins=2) == pytest.approx(0.0)
    # always 0.9 confident, 50% right -> 0.4
    assert expected_calibration_error([0.9] * 10, [True, False] * 5, n_bins=5) == pytest.approx(0.4)


def test_reliability_bins_are_equal_mass() -> None:
    rng = np.random.default_rng(0)
    conf = rng.random(100).tolist()
    bins = reliability_bins(conf, [True] * 100, n_bins=10)
    assert [b.count for b in bins] == [10] * 10
    assert all(b.lo <= b.mean_confidence <= b.hi for b in bins)
    means = [b.mean_confidence for b in bins]
    assert means == sorted(means)


def test_wrong_at_confidence() -> None:
    share, rate, n = wrong_at_confidence([0.95, 0.95, 0.5, 0.92], [False, True, False, True])
    assert n == 3
    assert share == pytest.approx(1 / 4)
    assert rate == pytest.approx(1 / 3)


def test_bootstrap_ci_contains_point_and_is_deterministic() -> None:
    y = np.array([1, 0] * 50)
    est = bootstrap_ci(lambda i: float(y[i].mean()), len(y), n_boot=500, seed=3)
    assert est.value == pytest.approx(0.5)
    assert est.lo < 0.5 < est.hi
    assert est.lo > 0.35
    assert est.hi < 0.65
    again = bootstrap_ci(lambda i: float(y[i].mean()), len(y), n_boot=500, seed=3)
    assert (again.lo, again.hi) == (est.lo, est.hi)


def test_paired_bootstrap_detects_clear_difference() -> None:
    a = np.ones(100)
    b = np.r_[np.ones(50), np.zeros(50)]
    d = paired_bootstrap(
        lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), 100, n_boot=500
    )
    assert d.diff == pytest.approx(0.5)
    assert d.significant
    assert d.p_value < 0.01
    same = paired_bootstrap(
        lambda i: float(a[i].mean()), lambda i: float(a[i].mean()), 100, n_boot=200
    )
    assert same.diff == 0.0
    assert not same.significant


def test_mcnemar_exact() -> None:
    a = np.array([True] * 10 + [False] * 0 + [True] * 5)
    b = np.array([False] * 10 + [True] * 0 + [True] * 5)
    r = mcnemar_exact(a, b)
    assert (r.a_only, r.b_only) == (10, 0)
    assert r.p_value == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(a, a).p_value == 1.0
