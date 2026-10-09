"""Convert raw public datasets into judge :class:`Item` objects."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from judgment_calls.datasets.sampling import approx_tokens
from judgment_calls.models import Item

SUMMEVAL_DIMS = ("coherence", "consistency", "fluency", "relevance")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def load_ragtruth(raw_dir: Path, split: str = "test") -> list[Item]:
    """RAGTruth responses -> binary hallucination items.

    Label: hallucinated iff human annotators marked at least one span. Responses whose
    `quality` is not "good" (incorrect refusals, truncations) are excluded.
    """
    sources: dict[str, dict[str, Any]] = {}
    with (raw_dir / "ragtruth_source_info.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            sources[row["source_id"]] = row
    items: list[Item] = []
    with (raw_dir / "ragtruth_response.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] != split or row["quality"] != "good":
                continue
            src = sources[row["source_id"]]
            state = {"response": row["response"], "source": src["prompt"]}
            labels = row["labels"]
            items.append(
                Item(
                    task="faithfulness",
                    item_id=f"ragtruth-{row['id']}",
                    state=state,
                    gold={"hallucination": len(labels) > 0},
                    meta={
                        "dataset": "RAGTruth",
                        "subtask": src["task_type"],
                        "generator": row["model"],
                        "n_spans": len(labels),
                        "span_types": sorted({lab.get("label_type", "") for lab in labels}),
                        "approx_tokens": approx_tokens(state),
                    },
                )
            )
    return items


def load_summeval(raw_dir: Path) -> list[Item]:
    """SummEval (mteb copy) -> one item per (article, machine summary).

    Gold per dimension: mean of the three expert ratings, shifted to the 0-based
    level-index scale (1..5 -> 0..4).
    """
    items: list[Item] = []
    for row in _read_parquet_rows(raw_dir / "summeval.parquet"):
        summaries: list[str] = row["machine_summaries"]
        for j, summary in enumerate(summaries):
            state = {"summary": summary, "article": row["text"]}
            gold: dict[str, bool | str | float] = {d: float(row[d][j]) - 1.0 for d in SUMMEVAL_DIMS}
            items.append(
                Item(
                    task="summeval",
                    item_id=f"summeval-{row['id']}-{j:02d}",
                    state=state,
                    gold=gold,
                    meta={
                        "dataset": "SummEval",
                        "doc_id": row["id"],
                        "approx_tokens": approx_tokens(state),
                    },
                )
            )
    return items


_WINNER = {"model_a": "A", "model_b": "B", "tie": "tie"}


def _mtbench_state(
    conv_a: list[dict[str, str]], conv_b: list[dict[str, str]], turn: int
) -> dict[str, str]:
    if turn == 1:
        return {
            "user_message": conv_a[0]["content"],
            "answer_A": conv_a[1]["content"],
            "answer_B": conv_b[1]["content"],
        }
    return {
        "first_user_message": conv_a[0]["content"],
        "first_answer_A": conv_a[1]["content"],
        "first_answer_B": conv_b[1]["content"],
        "follow_up_user_message": conv_a[2]["content"],
        "final_answer_A": conv_a[3]["content"],
        "final_answer_B": conv_b[3]["content"],
    }


def load_mtbench(raw_dir: Path) -> tuple[list[Item], dict[str, Any]]:
    """MT-Bench human judgments -> one item per (question, model pair, turn).

    Gold: strict majority of expert votes (A / B / tie); comparisons without a strict
    majority are dropped. Also returns human-human agreement on multiply-voted pairs,
    which serves as a ceiling for judge-human agreement.
    """
    groups: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw_row in _read_parquet_rows(raw_dir / "mtbench_human.parquet"):
        row = dict(raw_row)
        ma, mb = str(row["model_a"]), str(row["model_b"])
        if ma > mb:  # canonical model order so votes on the same pair are pooled
            row["model_a"], row["model_b"] = mb, ma
            row["conversation_a"], row["conversation_b"] = (
                row["conversation_b"],
                row["conversation_a"],
            )
            row["winner"] = {"model_a": "model_b", "model_b": "model_a"}.get(
                str(row["winner"]), str(row["winner"])
            )
        key = (int(row["question_id"]), str(row["model_a"]), str(row["model_b"]), int(row["turn"]))
        groups[key].append(row)

    items: list[Item] = []
    agree_pairs = 0
    total_pairs = 0
    agree_pairs_no_tie = 0
    total_pairs_no_tie = 0
    dropped = 0
    for (qid, ma, mb, turn), rows in sorted(groups.items()):
        votes = [_WINNER[str(r["winner"])] for r in rows]
        for v1, v2 in combinations(votes, 2):
            total_pairs += 1
            agree_pairs += int(v1 == v2)
            if v1 != "tie" and v2 != "tie":
                total_pairs_no_tie += 1
                agree_pairs_no_tie += int(v1 == v2)
        counts = Counter(votes).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            dropped += 1
            continue
        label = counts[0][0]
        state = _mtbench_state(
            list(rows[0]["conversation_a"]), list(rows[0]["conversation_b"]), turn
        )
        items.append(
            Item(
                task="pairwise",
                item_id=f"mtbench-q{qid}-t{turn}-{ma}-vs-{mb}",
                state=state,
                gold={"preference": label},
                meta={
                    "dataset": "MT-Bench human judgments",
                    "question_id": qid,
                    "turn": turn,
                    "model_a": ma,
                    "model_b": mb,
                    "n_votes": len(votes),
                    "votes": dict(Counter(votes)),
                    "approx_tokens": approx_tokens(state),
                },
            )
        )
    stats: dict[str, Any] = {
        "comparisons": len(groups),
        "dropped_no_majority": dropped,
        "multi_vote_comparisons": sum(1 for r in groups.values() if len(r) > 1),
        "human_pair_agreement_with_ties": agree_pairs / total_pairs if total_pairs else None,
        "human_pairs_with_ties": total_pairs,
        "human_pair_agreement_without_ties": (
            agree_pairs_no_tie / total_pairs_no_tie if total_pairs_no_tie else None
        ),
        "human_pairs_without_ties": total_pairs_no_tie,
    }
    return items, stats
