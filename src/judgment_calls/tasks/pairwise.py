"""Pairwise preference with explicit ties (`choice` question), judged in both orders."""

from __future__ import annotations

from judgment_calls.models import Item, QuestionSpec, TaskSpec, Variant

SPEC = TaskSpec(
    task="pairwise",
    version="1.0",
    description=(
        "Compare two assistants' answers to the same user turn and pick A, B or tie. "
        "Human labels: MT-Bench expert pairwise votes (majority per comparison). Every item "
        "is judged twice, with A/B swapped, to measure position bias."
    ),
    questions=[
        QuestionSpec(
            name="preference",
            kind="choice",
            instructions=(
                "Two AI assistants, A and B, answered the same user. Judge which assistant "
                "gave the better answer to the user's final message, considering helpfulness, "
                "relevance, accuracy, depth, creativity and level of detail. For a follow-up "
                "question, judge the final answer in the context of that assistant's own "
                "conversation. Do not let the order of the answers, their length or the "
                "assistants' names influence you. Choose tie only if the two answers are of "
                "about equal quality."
            ),
            criteria={
                "A": "Assistant A's answer is better.",
                "B": "Assistant B's answer is better.",
                "tie": "Both answers are of about equal quality.",
            },
        )
    ],
)

_SWAP = {"A": "B", "B": "A", "tie": "tie"}


def swap_state(state: dict[str, str]) -> dict[str, str]:
    """Swap every *_A / *_B field pair while keeping key order."""
    out: dict[str, str] = {}
    for key, value in state.items():
        if key.endswith("_A"):
            out[key] = state[key[:-2] + "_B"]
        elif key.endswith("_B"):
            out[key] = state[key[:-2] + "_A"]
        else:
            out[key] = value
    return out


def swap_label(label: str) -> str:
    return _SWAP[label]


def variants(item: Item) -> list[Variant]:
    gold = item.gold["preference"]
    assert isinstance(gold, str)
    return [
        Variant(name="orig", state=item.state, gold=item.gold),
        Variant(name="swap", state=swap_state(item.state), gold={"preference": swap_label(gold)}),
    ]
