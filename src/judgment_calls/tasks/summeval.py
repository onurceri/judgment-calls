"""Rubric scoring of summaries on four SummEval dimensions (`score` questions)."""

from __future__ import annotations

from judgment_calls.models import Item, QuestionSpec, TaskSpec, Variant

LEVELS = ["1 - very poor", "2 - poor", "3 - acceptable", "4 - good", "5 - excellent"]

_DIMENSIONS: dict[str, str] = {
    "coherence": (
        "Rate the COHERENCE of the summary: the collective quality of all its sentences. "
        "A coherent summary is well-structured and well-organized and builds from sentence "
        "to sentence into a coherent body of information about the topic, rather than a "
        "heap of related facts."
    ),
    "consistency": (
        "Rate the CONSISTENCY of the summary: its factual alignment with the source article. "
        "A consistent summary contains only statements that are entailed by the article. "
        "Penalize summaries that contain hallucinated or contradicting facts."
    ),
    "fluency": (
        "Rate the FLUENCY of the summary: the quality of its individual sentences. "
        "Sentences should be grammatical, with correct spelling, punctuation, word choice "
        "and sentence structure, and easy to read."
    ),
    "relevance": (
        "Rate the RELEVANCE of the summary: how well it selects the important content of "
        "the article. It should include only important information and should be penalized "
        "for redundancies and excess or unimportant information."
    ),
}

SPEC = TaskSpec(
    task="summeval",
    version="1.0",
    description=(
        "Rate a news summary against its source article on coherence, consistency, fluency "
        "and relevance (1-5). Human labels: SummEval expert annotations (mean of 3 experts)."
    ),
    questions=[
        QuestionSpec(name=name, kind="score", instructions=text, criteria=list(LEVELS))
        for name, text in _DIMENSIONS.items()
    ],
)


def variants(item: Item) -> list[Variant]:
    return [Variant(name="orig", state=item.state, gold=item.gold)]
