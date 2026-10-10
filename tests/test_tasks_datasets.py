from __future__ import annotations

from pathlib import Path

from judgment_calls.datasets.sampling import (
    allocate,
    approx_tokens,
    length_bucket,
    stratified_sample,
)
from judgment_calls.models import Item
from judgment_calls.report.analysis import holm
from judgment_calls.tasks import TASKS, pairwise


def test_specs_are_versioned_and_hash_stable() -> None:
    for tdef in TASKS.values():
        assert tdef.spec.version
        assert tdef.spec.spec_hash() == tdef.spec.spec_hash()
        assert len(tdef.spec.spec_hash()) == 16


def test_pairwise_swap_roundtrip() -> None:
    state = {
        "first_user_message": "q",
        "first_answer_A": "a1",
        "first_answer_B": "b1",
        "final_answer_A": "a2",
        "final_answer_B": "b2",
    }
    item = Item(task="pairwise", item_id="x", state=state, gold={"preference": "A"})
    orig, swap = pairwise.variants(item)
    assert swap.state["first_answer_A"] == "b1"
    assert swap.state["final_answer_B"] == "a2"
    assert list(swap.state) == list(state)  # key order preserved
    assert swap.gold == {"preference": "B"}
    assert pairwise.swap_state(swap.state) == state
    assert pairwise.swap_label("tie") == "tie"
    assert orig.gold == item.gold


def test_allocate_water_fills() -> None:
    assert allocate(10, [5, 5]) == [5, 5]
    assert allocate(10, [2, 10, 10]) == [2, 4, 4]
    assert allocate(7, [100, 100]) == [4, 3]
    assert allocate(50, [3, 4]) == [3, 4]
    assert sum(allocate(200, [6, 43, 778, 604, 169])) == 200


def test_length_bucket() -> None:
    assert length_bucket(10) == 0
    assert length_bucket(700) == 1
    assert length_bucket(5000) == 3
    assert approx_tokens({"a": "x" * 400}) == 100


def _items(n: int) -> list[Item]:
    return [
        Item(
            task="t",
            item_id=f"i{k:03d}",
            state={"s": "x" * (100 * (k % 40))},
            gold={"q": k % 3 == 0},
        )
        for k in range(n)
    ]


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    items = _items(300)
    a = stratified_sample(items, 60, lambda it: it.gold["q"], seed=7)
    b = stratified_sample(items, 60, lambda it: it.gold["q"], seed=7)
    assert [x.item_id for x in a] == [x.item_id for x in b]
    assert len(a) == 60
    assert sum(1 for x in a if x.gold["q"]) == 30
    assert len({x.item_id for x in a}) == 60


def test_holm() -> None:
    adj = holm([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]


def test_reference_subset_roundtrip(tmp_path: Path) -> None:
    from judgment_calls.datasets.prepare import write_reference_subset
    from judgment_calls.datasets.store import write_items
    from judgment_calls.runner import read_subset

    items = [
        Item(
            task="faithfulness",
            item_id=f"f{k:03d}",
            state={"s": "x" * (50 * k)},
            gold={"hallucination": k % 2 == 0},
        )
        for k in range(80)
    ]
    write_items(tmp_path / "faithfulness.jsonl", items)
    out = write_reference_subset(tmp_path, tmp_path / "sub", {"faithfulness": 20}, seed=1)
    assert out["faithfulness"]["labels"] == {"False": 10, "True": 10}
    ids = read_subset(tmp_path / "sub" / "faithfulness.txt")
    assert len(ids) == 20
    again = write_reference_subset(tmp_path, tmp_path / "sub2", {"faithfulness": 20}, seed=1)
    assert again == out
    assert read_subset(tmp_path / "sub2" / "faithfulness.txt") == ids
