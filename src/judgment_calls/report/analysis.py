"""Per judge x task metrics with bootstrap CIs, paired comparisons and diagnostics."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from judgment_calls.datasets.sampling import DEFAULT_LENGTH_EDGES, bucket_label, length_bucket
from judgment_calls.judges import JUDGES_BY_ID
from judgment_calls.metrics.agreement import (
    accuracy,
    balanced_accuracy,
    cohen_kappa,
    macro_f1,
)
from judgment_calls.metrics.calibration import (
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    wrong_at_confidence,
)
from judgment_calls.metrics.ranking import auroc, kendall_tau, mae, spearman
from judgment_calls.metrics.uncertainty import (
    Estimate,
    PairedDiff,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap,
)
from judgment_calls.models import Answer, JudgmentRecord
from judgment_calls.report.load import TaskResults
from judgment_calls.tasks.pairwise import swap_label

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

N_BOOT = 2000
MIN_N = 30  # judges with fewer judged items are listed as smoke-only
IntArr = NDArray[np.int64]
FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]


@dataclass(frozen=True)
class View:
    """A judge plus the confidence signal being analysed."""

    judge_id: str
    signal: str  # "native" (decision prob / verbalized) or "logprob"

    @property
    def label(self) -> str:
        return self.judge_id if self.signal == "native" else f"{self.judge_id} (logprob)"

    @property
    def kind(self) -> str:
        cfg = JUDGES_BY_ID.get(self.judge_id)
        return cfg.kind if cfg else "?"


def views_for(tr: TaskResults) -> list[View]:
    out: list[View] = []
    for j in tr.judges():
        out.append(View(j, "native"))
        recs = tr.for_judge(j).values()
        if any(a.logprob_probabilities for r in recs for a in r.answers.values()):
            out.append(View(j, "logprob"))
    return out


def _dist(view: View, ans: Answer) -> dict[str, float] | None:
    if view.signal == "logprob":
        return ans.logprob_probabilities
    return ans.probabilities


def _pred_conf(view: View, ans: Answer) -> tuple[str, float]:
    """Prediction and top-label confidence for a view."""
    dist = _dist(view, ans)
    if view.signal == "logprob" and dist:
        pred = max(dist, key=lambda k: dist[k])
        return pred, dist[pred]
    if dist:
        return ans.prediction, dist.get(ans.prediction, 0.0)
    conf = ans.confidence if ans.confidence is not None else 0.5
    return ans.prediction, conf


# ---------------------------------------------------------------------------
# generic containers


@dataclass
class CalibrationSummary:
    ece: Estimate
    brier: Estimate
    wrong_at_90: float
    error_rate_at_90: float
    n_at_90: int
    bins: list[ReliabilityBin]
    mean_confidence: float


@dataclass
class LengthRow:
    bucket: str
    n: int
    metric: Estimate


@dataclass
class ViewResult:
    view: View
    n: int
    metrics: dict[str, Estimate] = field(default_factory=dict[str, Estimate])
    extra: dict[str, float] = field(default_factory=dict[str, float])
    calibration: CalibrationSummary | None = None
    by_length: list[LengthRow] = field(default_factory=list[LengthRow])
    by_group: dict[str, Estimate] = field(default_factory=dict[str, Estimate])


@dataclass
class Comparison:
    task: str
    metric: str
    a: str
    b: str
    n: int
    diff: PairedDiff
    mcnemar_p: float | None = None
    a_only: int | None = None
    b_only: int | None = None
    p_holm: float | None = None


def _ci(fn: Callable[[IntArr], float], n: int, seed: int = 0) -> Estimate:
    return bootstrap_ci(fn, n, n_boot=N_BOOT, seed=seed)


def _calibration(
    conf: FloatArr, correct: BoolArr, prob_for_brier: FloatArr, outcome_for_brier: BoolArr
) -> CalibrationSummary:
    n = len(conf)
    w90, e90, n90 = wrong_at_confidence(conf.tolist(), correct.tolist(), 0.9)
    return CalibrationSummary(
        ece=_ci(lambda i: expected_calibration_error(conf[i].tolist(), correct[i].tolist()), n),
        brier=_ci(
            lambda i: brier_score(prob_for_brier[i].tolist(), outcome_for_brier[i].tolist()), n
        ),
        wrong_at_90=w90,
        error_rate_at_90=e90,
        n_at_90=n90,
        bins=reliability_bins(conf.tolist(), correct.tolist(), 10),
        mean_confidence=float(conf.mean()) if n else float("nan"),
    )


def _bucket(tr: TaskResults, item_id: str) -> int:
    return length_bucket(int(tr.items[item_id].meta.get("approx_tokens", 0)))


# ---------------------------------------------------------------------------
# faithfulness (noul)


@dataclass
class NoulData:
    item_ids: list[str]
    y: BoolArr
    pred: BoolArr
    p_true: FloatArr
    conf: FloatArr


def noul_data(tr: TaskResults, view: View, qname: str = "hallucination") -> NoulData:
    rows = sorted(tr.for_judge(view.judge_id).items())
    ids: list[str] = []
    y: list[bool] = []
    pred: list[bool] = []
    p_true: list[float] = []
    conf: list[float] = []
    for (item_id, _variant), rec in rows:
        ans = rec.answers.get(qname)
        if ans is None:
            continue
        dist = _dist(view, ans)
        if view.signal == "logprob" and not dist:
            continue
        label, c = _pred_conf(view, ans)
        is_true = label == "true"
        pt = dist["true"] if dist else (c if is_true else 1.0 - c)
        ids.append(item_id)
        y.append(bool(rec.gold[qname]))
        pred.append(is_true)
        p_true.append(pt)
        conf.append(c)
    return NoulData(
        ids,
        np.array(y, dtype=bool),
        np.array(pred, dtype=bool),
        np.array(p_true, dtype=np.float64),
        np.array(conf, dtype=np.float64),
    )


def analyse_noul(tr: TaskResults, view: View) -> ViewResult:
    d = noul_data(tr, view)
    n = len(d.y)
    res = ViewResult(view, n)
    if n == 0:
        return res
    correct = d.y == d.pred
    yl, pl = d.y.tolist(), d.pred.tolist()
    res.metrics["accuracy"] = _ci(lambda i: accuracy(d.y[i].tolist(), d.pred[i].tolist()), n)
    res.metrics["balanced_accuracy"] = _ci(
        lambda i: balanced_accuracy(d.y[i].tolist(), d.pred[i].tolist(), [True, False]), n
    )
    res.metrics["macro_f1"] = _ci(
        lambda i: macro_f1(d.y[i].tolist(), d.pred[i].tolist(), [True, False]), n
    )
    res.metrics["kappa"] = _ci(
        lambda i: cohen_kappa(d.y[i].tolist(), d.pred[i].tolist(), [True, False]), n
    )
    res.metrics["auroc"] = _ci(lambda i: auroc(d.y[i].tolist(), d.p_true[i].tolist()), n)
    tp = float(np.sum(d.pred & d.y))
    res.extra["recall_halluc"] = tp / max(1.0, float(d.y.sum()))
    res.extra["precision_halluc"] = tp / max(1.0, float(d.pred.sum()))
    res.extra["flagged_rate"] = float(d.pred.mean())
    res.extra["accuracy_point"] = accuracy(yl, pl)
    res.calibration = _calibration(d.conf, correct, d.p_true, d.y)
    buckets = np.array([_bucket(tr, i) for i in d.item_ids])
    for b in sorted(set(buckets.tolist())):
        idx = np.where(buckets == b)[0]
        yb, pb = d.y[idx], d.pred[idx]
        res.by_length.append(
            LengthRow(
                bucket_label(b, DEFAULT_LENGTH_EDGES),
                len(idx),
                _ci(
                    lambda i, yb=yb, pb=pb: balanced_accuracy(
                        yb[i].tolist(), pb[i].tolist(), [True, False]
                    ),
                    len(idx),
                ),
            )
        )
    groups: dict[str, list[int]] = defaultdict(list)
    for k, item_id in enumerate(d.item_ids):
        groups[str(tr.items[item_id].meta.get("subtask"))].append(k)
    for g, idx_l in sorted(groups.items()):
        idx = np.array(idx_l)
        yg, pg = d.y[idx], d.pred[idx]
        res.by_group[g] = _ci(
            lambda i, yg=yg, pg=pg: accuracy(yg[i].tolist(), pg[i].tolist()), len(idx)
        )
    return res


# ---------------------------------------------------------------------------
# summeval (score)

SUMMEVAL_DIMS = ("coherence", "consistency", "fluency", "relevance")


@dataclass
class ScoreData:
    item_ids: list[str]
    gold: FloatArr  # (n, d) level-index scale
    expected: FloatArr  # (n, d)


def score_data(tr: TaskResults, view: View, dims: tuple[str, ...] = SUMMEVAL_DIMS) -> ScoreData:
    ids: list[str] = []
    gold: list[list[float]] = []
    exp: list[list[float]] = []
    for (item_id, _v), rec in sorted(tr.for_judge(view.judge_id).items()):
        row_g: list[float] = []
        row_e: list[float] = []
        for dname in dims:
            ans = rec.answers.get(dname)
            if ans is None:
                break
            if view.signal == "logprob":
                dist = ans.logprob_probabilities
                if not dist:
                    break
                e = sum(int(k) * v for k, v in dist.items())
            else:
                e = ans.expected_score if ans.expected_score is not None else float(ans.prediction)
            row_g.append(float(rec.gold[dname]))
            row_e.append(e)
        else:
            ids.append(item_id)
            gold.append(row_g)
            exp.append(row_e)
    return ScoreData(
        ids,
        np.array(gold, dtype=np.float64).reshape(-1, len(dims)),
        np.array(exp, dtype=np.float64).reshape(-1, len(dims)),
    )


def _qwk(g: FloatArr, e: FloatArr, levels: int = 5) -> float:
    gi = np.clip(np.rint(g), 0, levels - 1).astype(int).tolist()
    ei = np.clip(np.rint(e), 0, levels - 1).astype(int).tolist()
    return cohen_kappa(gi, ei, list(range(levels)), weights="quadratic")


def analyse_score(tr: TaskResults, view: View) -> ViewResult:
    d = score_data(tr, view)
    n = len(d.item_ids)
    res = ViewResult(view, n)
    if n == 0:
        return res
    k = d.gold.shape[1]

    def mean_rho(i: IntArr) -> float:
        return float(
            np.nanmean(
                [spearman(d.expected[i, j].tolist(), d.gold[i, j].tolist()) for j in range(k)]
            )
        )

    res.metrics["mean_spearman"] = _ci(mean_rho, n)
    res.metrics["mean_kendall"] = _ci(
        lambda i: float(
            np.nanmean(
                [kendall_tau(d.expected[i, j].tolist(), d.gold[i, j].tolist()) for j in range(k)]
            )
        ),
        n,
    )
    res.metrics["mean_qwk"] = _ci(
        lambda i: float(np.mean([_qwk(d.gold[i, j], d.expected[i, j]) for j in range(k)])), n
    )
    res.metrics["mean_mae"] = _ci(
        lambda i: float(
            np.mean([mae(d.expected[i, j].tolist(), d.gold[i, j].tolist()) for j in range(k)])
        ),
        n,
    )
    for j, dname in enumerate(SUMMEVAL_DIMS):
        res.metrics[f"spearman_{dname}"] = _ci(
            lambda i, j=j: spearman(d.expected[i, j].tolist(), d.gold[i, j].tolist()), n
        )
        res.extra[f"qwk_{dname}"] = _qwk(d.gold[:, j], d.expected[:, j])
        res.extra[f"mae_{dname}"] = mae(d.expected[:, j].tolist(), d.gold[:, j].tolist())
        res.extra[f"distinct_{dname}"] = float(len(np.unique(np.round(d.expected[:, j], 3))))
    res.extra["bias"] = float(np.mean(d.expected - d.gold))
    buckets = np.array([_bucket(tr, i) for i in d.item_ids])
    for b in sorted(set(buckets.tolist())):
        idx = np.where(buckets == b)[0]
        gb, eb = d.gold[idx], d.expected[idx]
        res.by_length.append(
            LengthRow(
                bucket_label(b, DEFAULT_LENGTH_EDGES),
                len(idx),
                _ci(
                    lambda i, gb=gb, eb=eb: float(
                        np.nanmean(
                            [spearman(eb[i, j].tolist(), gb[i, j].tolist()) for j in range(k)]
                        )
                    ),
                    len(idx),
                ),
            )
        )
    return res


# ---------------------------------------------------------------------------
# pairwise (choice, both orders)

CHOICE_LABELS = ["A", "B", "tie"]


@dataclass
class PairData:
    item_ids: list[str]
    gold: list[str]  # gold in original order
    pred_orig: list[str | None]
    pred_swap_mapped: list[str | None]  # swapped-order prediction mapped back to orig labels
    pred_swap_raw: list[str | None]
    conf: list[float]  # per judgment (flattened)
    correct: list[bool]  # per judgment (flattened)


def pair_data(tr: TaskResults, view: View, qname: str = "preference") -> PairData:
    recs = tr.for_judge(view.judge_id)
    ids = sorted({i for (i, _v) in recs})
    out = PairData([], [], [], [], [], [], [])
    for item_id in ids:
        gold = tr.items[item_id].gold[qname]
        assert isinstance(gold, str)
        preds: dict[str, str | None] = {"orig": None, "swap": None}
        for variant in ("orig", "swap"):
            rec = recs.get((item_id, variant))
            if rec is None or qname not in rec.answers:
                continue
            ans = rec.answers[qname]
            if view.signal == "logprob" and not ans.logprob_probabilities:
                continue
            p, c = _pred_conf(view, ans)
            preds[variant] = p
            out.conf.append(c)
            out.correct.append(p == rec.gold[qname])
        if preds["orig"] is None and preds["swap"] is None:
            continue
        out.item_ids.append(item_id)
        out.gold.append(gold)
        out.pred_orig.append(preds["orig"])
        out.pred_swap_raw.append(preds["swap"])
        out.pred_swap_mapped.append(swap_label(preds["swap"]) if preds["swap"] else None)
    return out


def _ints(i: IntArr) -> list[int]:
    return [int(x) for x in i]


def _item_acc(d: PairData, i: IntArr) -> float:
    vals: list[float] = []
    for k in _ints(i):
        js = [p for p in (d.pred_orig[k], d.pred_swap_mapped[k]) if p is not None]
        vals.append(float(np.mean([p == d.gold[k] for p in js])))
    return float(np.mean(vals)) if vals else float("nan")


def _flat(d: PairData, i: IntArr) -> tuple[list[str], list[str]]:
    g: list[str] = []
    p: list[str] = []
    for k in _ints(i):
        for pred in (d.pred_orig[k], d.pred_swap_mapped[k]):
            if pred is not None:
                g.append(d.gold[k])
                p.append(pred)
    return g, p


def _agree_no_tie(d: PairData, i: IntArr) -> float:
    g, p = _flat(d, i)
    pairs = [(a, b) for a, b in zip(g, p, strict=True) if a != "tie" and b != "tie"]
    return float(np.mean([a == b for a, b in pairs])) if pairs else float("nan")


@dataclass
class PositionBias:
    n_items: int
    consistent: float
    favors_first: float
    favors_second: float
    other_inconsistent: float
    first_position_rate: float  # share of non-tie judgments that pick the first-shown answer


def position_bias(d: PairData) -> PositionBias:
    both = [
        k
        for k in range(len(d.item_ids))
        if d.pred_orig[k] is not None and d.pred_swap_raw[k] is not None
    ]
    n = len(both)
    cons = first = second = other = 0
    first_pos = nontie = 0
    for k in both:
        o, s_raw, s_map = d.pred_orig[k], d.pred_swap_raw[k], d.pred_swap_mapped[k]
        if o == s_map:
            cons += 1
        elif o == "A" and s_raw == "A":
            first += 1
        elif o == "B" and s_raw == "B":
            second += 1
        else:
            other += 1
        for p in (o, s_raw):
            if p != "tie":
                nontie += 1
                first_pos += int(p == "A")
    f = float(max(n, 1))
    return PositionBias(
        n,
        cons / f,
        first / f,
        second / f,
        other / f,
        first_pos / nontie if nontie else float("nan"),
    )


def analyse_pairwise(tr: TaskResults, view: View) -> tuple[ViewResult, PositionBias]:
    d = pair_data(tr, view)
    n = len(d.item_ids)
    res = ViewResult(view, n)
    if n == 0:
        return res, position_bias(d)
    res.metrics["accuracy"] = _ci(lambda i: _item_acc(d, i), n)
    res.metrics["macro_f1"] = _ci(lambda i: macro_f1(*_flat(d, i), CHOICE_LABELS), n)
    res.metrics["kappa"] = _ci(lambda i: cohen_kappa(*_flat(d, i), CHOICE_LABELS), n)
    res.metrics["agreement_no_tie"] = _ci(lambda i: _agree_no_tie(d, i), n)
    pb = position_bias(d)
    res.metrics["consistency"] = _ci(
        lambda i: float(
            np.mean(
                [
                    d.pred_orig[k] == d.pred_swap_mapped[k]
                    for k in _ints(i)
                    if d.pred_orig[k] is not None and d.pred_swap_mapped[k] is not None
                ]
                or [np.nan]
            )
        ),
        n,
    )
    g_all, p_all = _flat(d, np.arange(n, dtype=np.int64))
    res.extra["tie_rate_pred"] = float(np.mean([p == "tie" for p in p_all]))
    res.extra["tie_rate_gold"] = float(np.mean([x == "tie" for x in g_all]))
    conf = np.array(d.conf, dtype=np.float64)
    correct = np.array(d.correct, dtype=bool)
    res.calibration = _calibration(conf, correct, conf, correct)
    buckets = np.array([_bucket(tr, i) for i in d.item_ids])
    for b in sorted(set(buckets.tolist())):
        idx = np.where(buckets == b)[0]
        res.by_length.append(
            LengthRow(
                bucket_label(b, DEFAULT_LENGTH_EDGES),
                len(idx),
                _ci(lambda i, idx=idx: _item_acc(d, idx[i]), len(idx)),
            )
        )
    for turn in (1, 2):
        idx = np.array(
            [k for k, it in enumerate(d.item_ids) if tr.items[it].meta.get("turn") == turn],
            dtype=np.int64,
        )
        if len(idx):
            res.by_group[f"turn {turn}"] = _ci(lambda i, idx=idx: _item_acc(d, idx[i]), len(idx))
    return res, pb


# ---------------------------------------------------------------------------
# consistency across repeats, cost and latency


@dataclass
class RepeatStats:
    judge_id: str
    task: str
    n_groups: int  # (item, variant) groups with all repeats
    n_answers: int
    flip_rate: float
    mean_std: float  # std of the confidence/probability (or expected score) across repeats


def repeat_stats(tr: TaskResults, judge_id: str, n_repeats: int = 3) -> RepeatStats | None:
    groups: dict[tuple[str, str], dict[int, JudgmentRecord]] = defaultdict(dict)
    for (j, item, variant, rep), rec in tr.records.items():
        if j == judge_id:
            groups[(item, variant)][rep] = rec
    full = [g for g in groups.values() if all(r in g for r in range(n_repeats))]
    if not full:
        return None
    flips: list[bool] = []
    stds: list[float] = []
    for g in full:
        for qname in g[0].answers:
            answers = [g[r].answers[qname] for r in range(n_repeats) if qname in g[r].answers]
            if len(answers) < n_repeats:
                continue
            flips.append(len({a.prediction for a in answers}) > 1)
            if answers[0].kind == "score":
                vals = [a.expected_score or 0.0 for a in answers]
            elif answers[0].probabilities:
                lab = answers[0].prediction
                vals = [(a.probabilities or {}).get(lab, 0.0) for a in answers]
            else:
                vals = [a.confidence if a.confidence is not None else 0.5 for a in answers]
            stds.append(float(np.std(vals)))
    return RepeatStats(
        judge_id,
        tr.task,
        len(full),
        len(flips),
        float(np.mean(flips)) if flips else float("nan"),
        float(np.mean(stds)) if stds else float("nan"),
    )


@dataclass
class CostLatency:
    judge_id: str
    n_calls: int
    total_cost: float
    cost_per_1k: float
    p50_s: float
    p95_s: float
    in_tokens_mean: float


def cost_latency(records: list[JudgmentRecord], judge_id: str) -> CostLatency:
    rs = [r for r in records if r.judge_id == judge_id]
    costs = [r.cost_usd for r in rs if r.cost_usd is not None]
    lat = [r.latency_s for r in rs if r.latency_s is not None]
    toks = [r.input_tokens for r in rs if r.input_tokens is not None]
    return CostLatency(
        judge_id,
        len(rs),
        float(sum(costs)),
        float(np.mean(costs) * 1000) if costs else float("nan"),
        float(np.percentile(lat, 50)) if lat else float("nan"),
        float(np.percentile(lat, 95)) if lat else float("nan"),
        float(np.mean(toks)) if toks else float("nan"),
    )


# ---------------------------------------------------------------------------
# paired comparisons


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def compare_noul(tr: TaskResults, a: View, b: View) -> list[Comparison]:
    da, db = noul_data(tr, a), noul_data(tr, b)
    common = sorted(set(da.item_ids) & set(db.item_ids))
    if len(common) < MIN_N:
        return []
    ia = np.array([da.item_ids.index(i) for i in common])
    ib = np.array([db.item_ids.index(i) for i in common])
    y = da.y[ia]
    pa, pb = da.pred[ia], db.pred[ib]
    sa, sb = da.p_true[ia], db.p_true[ib]
    n = len(common)
    out: list[Comparison] = []
    mc = mcnemar_exact(pa == y, pb == y)
    diff = paired_bootstrap(
        lambda i: balanced_accuracy(y[i].tolist(), pa[i].tolist(), [True, False]),
        lambda i: balanced_accuracy(y[i].tolist(), pb[i].tolist(), [True, False]),
        n,
        n_boot=N_BOOT,
    )
    out.append(
        Comparison(
            tr.task,
            "balanced_accuracy",
            a.label,
            b.label,
            n,
            diff,
            mc.p_value,
            mc.a_only,
            mc.b_only,
        )
    )
    diff_auc = paired_bootstrap(
        lambda i: auroc(y[i].tolist(), sa[i].tolist()),
        lambda i: auroc(y[i].tolist(), sb[i].tolist()),
        n,
        n_boot=N_BOOT,
    )
    out.append(Comparison(tr.task, "auroc", a.label, b.label, n, diff_auc))
    diff_brier = paired_bootstrap(
        lambda i: brier_score(sa[i].tolist(), y[i].tolist()),
        lambda i: brier_score(sb[i].tolist(), y[i].tolist()),
        n,
        n_boot=N_BOOT,
    )
    out.append(Comparison(tr.task, "brier (lower=better)", a.label, b.label, n, diff_brier))
    ca, cb = da.conf[ia], db.conf[ib]
    ka, kb = pa == y, pb == y
    diff_ece = paired_bootstrap(
        lambda i: expected_calibration_error(ca[i].tolist(), ka[i].tolist()),
        lambda i: expected_calibration_error(cb[i].tolist(), kb[i].tolist()),
        n,
        n_boot=N_BOOT,
    )
    out.append(Comparison(tr.task, "ece (lower=better)", a.label, b.label, n, diff_ece))
    return out


def compare_score(tr: TaskResults, a: View, b: View) -> list[Comparison]:
    da, db = score_data(tr, a), score_data(tr, b)
    common = sorted(set(da.item_ids) & set(db.item_ids))
    if len(common) < MIN_N:
        return []
    ia = np.array([da.item_ids.index(i) for i in common])
    ib = np.array([db.item_ids.index(i) for i in common])
    g, ea, eb = da.gold[ia], da.expected[ia], db.expected[ib]
    k = g.shape[1]

    def rho(e: FloatArr) -> Callable[[IntArr], float]:
        return lambda i: float(
            np.nanmean([spearman(e[i, j].tolist(), g[i, j].tolist()) for j in range(k)])
        )

    diff = paired_bootstrap(rho(ea), rho(eb), len(common), n_boot=N_BOOT)
    return [Comparison(tr.task, "mean_spearman", a.label, b.label, len(common), diff)]


def compare_pairwise(tr: TaskResults, a: View, b: View) -> list[Comparison]:
    da, db = pair_data(tr, a), pair_data(tr, b)
    common = sorted(set(da.item_ids) & set(db.item_ids))
    if len(common) < MIN_N:
        return []
    ia = np.array([da.item_ids.index(i) for i in common], dtype=np.int64)
    ib = np.array([db.item_ids.index(i) for i in common], dtype=np.int64)
    diff = paired_bootstrap(
        lambda i: _item_acc(da, ia[i]), lambda i: _item_acc(db, ib[i]), len(common), n_boot=N_BOOT
    )
    ca = np.array([da.pred_orig[k] == da.gold[k] for k in _ints(ia)], dtype=bool)
    cb = np.array([db.pred_orig[k] == db.gold[k] for k in _ints(ib)], dtype=bool)
    mc = mcnemar_exact(ca, cb)
    return [
        Comparison(
            tr.task,
            "accuracy",
            a.label,
            b.label,
            len(common),
            diff,
            mc.p_value,
            mc.a_only,
            mc.b_only,
        )
    ]
