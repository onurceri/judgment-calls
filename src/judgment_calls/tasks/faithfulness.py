"""Faithfulness / hallucination detection (binary, `noul` question)."""

from __future__ import annotations

from judgment_calls.models import Item, QuestionSpec, TaskSpec, Variant

SPEC = TaskSpec(
    task="faithfulness",
    version="1.0",
    description=(
        "Given a SOURCE and a RESPONSE generated from it, decide whether the RESPONSE "
        "contains a hallucination. Human labels: RAGTruth span annotations "
        "(any annotated span => hallucinated)."
    ),
    questions=[
        QuestionSpec(
            name="hallucination",
            kind="noul",
            instructions=(
                "The 'response' was written by an AI assistant that was told to rely only on "
                "the 'source' (the task prompt and its reference material). Decide whether the "
                "response contains a hallucination: any claim that contradicts the source, or "
                "any claim of fact that is not supported by the source. Ignore style and "
                "completeness; judge only whether the content is grounded in the source."
            ),
            criteria={
                "true": (
                    "The response contains at least one claim that contradicts the source or "
                    "is not supported by it."
                ),
                "false": "Every claim in the response is supported by the source.",
            },
        )
    ],
)


def variants(item: Item) -> list[Variant]:
    # The (short) response is placed before the (long) source so that judges which
    # truncate long inputs still see the text under evaluation. Same order for all judges.
    return [Variant(name="orig", state=item.state, gold=item.gold)]
