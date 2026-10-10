"""Judge registry. Prices are USD per million tokens as listed by OpenRouter on
2026-10-09 and are used only for pre-call budget reservations; reported costs come
from each response's ``usage.cost``."""

from __future__ import annotations

from judgment_calls.models import JudgeConfig

JUDGES: list[JudgeConfig] = [
    # --- decision models (POST /api/v1/systemone) -------------------------------
    JudgeConfig(
        id="jev-1.13",
        kind="decision",
        model="typesafe/jev-1.13",
        price_in_per_m=0.042,
        context_tokens=64_000,
    ),
    JudgeConfig(
        id="clef",
        kind="decision",
        model="cloudflare/clef",
        price_in_per_m=0.24,
        context_tokens=65_536,
        notes="Reads only about the first 2K tokens of text state.",
    ),
    JudgeConfig(
        id="gpt-6-luna-decisions",
        kind="decision",
        model="openai/gpt-6-luna-decisions",
        price_in_per_m=0.10,
        context_tokens=1_050_000,
    ),
    JudgeConfig(
        id="d1", kind="decision", model="liquid/d1", price_in_per_m=0.04, context_tokens=65_536
    ),
    JudgeConfig(
        id="clef-flash",
        kind="decision",
        model="cloudflare/clef-flash",
        price_in_per_m=0.021,
        context_tokens=65_536,
        tier="secondary",
    ),
    JudgeConfig(
        id="solar-decide-flash",
        kind="decision",
        model="upstage/solar-decide-flash",
        price_in_per_m=0.05,
        context_tokens=524_288,
        tier="secondary",
    ),
    JudgeConfig(
        id="pplx-decider-v1.1",
        kind="decision",
        model="perplexity/pplx-decider-v1.1-27b",
        price_in_per_m=0.02,
        context_tokens=262_144,
        tier="secondary",
    ),
    JudgeConfig(
        id="mercury-decide",
        kind="decision",
        model="inception/mercury-decide",
        price_in_per_m=0.02,
        context_tokens=65_536,
        tier="secondary",
    ),
    JudgeConfig(
        id="tev1-4b",
        kind="decision",
        model="togethercomputer/tev1-4b-experimental",
        price_in_per_m=0.042,
        context_tokens=32_768,
        tier="secondary",
    ),
    JudgeConfig(
        id="kev-4b",
        kind="decision",
        model="jaredpalmer/kev-4b",
        price_in_per_m=0.042,
        context_tokens=8_192,
        tier="secondary",
    ),
    # --- generative judges (POST /api/v1/chat/completions, structured output) ----
    JudgeConfig(
        id="claude-haiku-5.5",
        kind="generative",
        model="anthropic/claude-haiku-5.5",
        price_in_per_m=0.10,
        price_out_per_m=0.50,
        extra_body={"reasoning": {"effort": "none"}},
        notes="Adaptive thinking disabled (it otherwise exhausted the output budget).",
    ),
    JudgeConfig(
        id="gpt-6-luna",
        kind="generative",
        model="openai/gpt-6-luna",
        price_in_per_m=0.10,
        price_out_per_m=0.50,
        extra_body={"reasoning": {"effort": "none"}},
        notes="Same base model as gpt-6-luna-decisions; hidden reasoning disabled.",
    ),
    JudgeConfig(
        id="gpt-4o-mini",
        kind="generative",
        model="openai/gpt-4o-mini",
        price_in_per_m=0.15,
        price_out_per_m=0.60,
        logprobs=True,
        tier="secondary",
        notes="Included because it exposes token logprobs (the two primary "
        "generative judges do not).",
    ),
    JudgeConfig(
        id="claude-sonnet-5.5",
        kind="generative",
        model="anthropic/claude-sonnet-5.5",
        price_in_per_m=2.0,
        price_out_per_m=10.0,
        tier="reference",
        # This endpoint rejects disabled reasoning; "minimal" produced 0 reasoning tokens in
        # probes. The output cap leaves room for a little reasoning plus the JSON answer.
        extra_body={"reasoning": {"effort": "minimal"}, "max_tokens": 400},
        notes="Stronger reference judge (reasoning is mandatory; effort=minimal); "
        "run on a fixed stratified subset only.",
    ),
]

JUDGES_BY_ID: dict[str, JudgeConfig] = {j.id: j for j in JUDGES}


def get_judges(ids: list[str] | None) -> list[JudgeConfig]:
    if not ids:
        return [j for j in JUDGES if j.tier == "primary"]
    out: list[JudgeConfig] = []
    for i in ids:
        if i in ("all-decision", "all-generative"):
            kind = i.split("-", 1)[1]
            out.extend(j for j in JUDGES if j.kind == kind and j.tier != "reference")
        elif i == "primary":
            out.extend(j for j in JUDGES if j.tier == "primary")
        else:
            try:
                out.append(JUDGES_BY_ID[i])
            except KeyError as exc:
                raise KeyError(f"unknown judge {i!r}; known: {sorted(JUDGES_BY_ID)}") from exc
    seen: set[str] = set()
    return [j for j in out if not (j.id in seen or seen.add(j.id))]
