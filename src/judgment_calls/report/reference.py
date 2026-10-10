"""Reference-judge analysis on a fixed subset (all judges compared on the same items)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from judgment_calls.judges import JUDGES
from judgment_calls.metrics.uncertainty import Estimate, PairedDiff, paired_bootstrap
from judgment_calls.report.analysis import (
    N_BOOT,
    Comparison,
    View,
    ViewResult,
    _qwk,  # pyright: ignore[reportPrivateUsage]
    analyse_noul,
    analyse_score,
    compare_noul,
    compare_score,
    holm,
    score_data,
)
from judgment_calls.report.load import TaskResults
from judgment_calls.runner import read_subset

REFERENCE_ID = "claude-sonnet-5.5"
FULL_RUN = [j.id for j in JUDGES if j.tier == "primary"]


@dataclass
class ReferenceTask:
    task: str
    n_subset: int
    n_done: int
    labels: dict[str, int]
    results: dict[str, ViewResult] = field(default_factory=dict[str, ViewResult])
    comparisons: list[Comparison] = field(default_factory=list[Comparison])


def _restrict(tr: TaskResults, ids: set[str]) -> TaskResults:
    return TaskResults(
        tr.task,
        {k: v for k, v in tr.items.items() if k in ids},
        {k: r for k, r in tr.records.items() if k[1] in ids},
        [],
    )


def _compare_qwk(tr: TaskResults, a: View, b: View) -> Comparison:
    da, db = score_data(tr, a), score_data(tr, b)
    common = sorted(set(da.item_ids) & set(db.item_ids))
    ia = np.array([da.item_ids.index(i) for i in common])
    ib = np.array([db.item_ids.index(i) for i in common])
    g, ea, eb = da.gold[ia], da.expected[ia], db.expected[ib]
    k = g.shape[1]
    diff: PairedDiff = paired_bootstrap(
        lambda i: float(np.mean([_qwk(g[i, j], ea[i, j]) for j in range(k)])),
        lambda i: float(np.mean([_qwk(g[i, j], eb[i, j]) for j in range(k)])),
        len(common),
        n_boot=N_BOOT,
    )
    return Comparison(tr.task, "mean_qwk", a.label, b.label, len(common), diff)


def analyse_reference(trs: dict[str, TaskResults], subset_dir: Path) -> list[ReferenceTask]:
    out: list[ReferenceTask] = []
    ref = View(REFERENCE_ID, "native")
    for task in ("faithfulness", "summeval"):
        path = subset_dir / f"{task}.txt"
        if task not in trs or not path.exists():
            continue
        subset = read_subset(path)
        tr = trs[task]
        done = {k[1] for k in tr.records if k[0] == REFERENCE_ID and k[3] == 0} & subset
        # only items every full-run judge has judged too
        for j in FULL_RUN:
            done &= {k[1] for k in tr.records if k[0] == j and k[3] == 0}
        sub = _restrict(tr, done)
        if task == "faithfulness":
            labels = {
                str(b): sum(1 for i in done if sub.items[i].gold["hallucination"] == b)
                for b in (True, False)
            }
        else:
            labels = {}
        rt = ReferenceTask(task, len(subset), len(done), labels)
        if not done:
            out.append(rt)
            continue
        for jid in [REFERENCE_ID, *FULL_RUN]:
            v = View(jid, "native")
            rt.results[jid] = (
                analyse_noul(sub, v) if task == "faithfulness" else analyse_score(sub, v)
            )
        family: list[Comparison] = []
        for jid in FULL_RUN:
            other = View(jid, "native")
            if task == "faithfulness":
                family.extend(compare_noul(sub, ref, other))
            else:
                family.extend(compare_score(sub, ref, other))
                family.append(_compare_qwk(sub, ref, other))
        for metric in sorted({c.metric for c in family}):
            fam = [c for c in family if c.metric == metric]
            for c, p in zip(fam, holm([c.diff.p_value for c in fam]), strict=True):
                c.p_holm = p
        rt.comparisons = family
        out.append(rt)
    return out


def ci_width(e: Estimate) -> float:
    return e.hi - e.lo
