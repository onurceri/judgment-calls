"""OpenRouter decision models via ``POST /api/v1/systemone``."""

from __future__ import annotations

import json
from typing import Any, cast

from judgment_calls.models import Answer, JudgeConfig, QuestionSpec, TaskSpec
from judgment_calls.providers.base import (
    OPENROUTER_BASE,
    ParsedResponse,
    ParseError,
    estimate_tokens,
)


def question_payload(q: QuestionSpec) -> dict[str, Any]:
    criteria: dict[str, str] | list[str] = (
        list(q.criteria) if isinstance(q.criteria, list) else dict(q.criteria)
    )
    return {"type": q.kind, "instructions": q.instructions, "criteria": criteria}


def _f(x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise ParseError(f"expected a number, got {x!r}")
    return float(x)


def parse_answer(q: QuestionSpec, raw: dict[str, Any]) -> Answer:
    if q.kind == "noul":
        p = _f(raw.get("noul"))
        if not 0.0 <= p <= 1.0:
            raise ParseError(f"noul out of range: {p}")
        pred = "true" if p >= 0.5 else "false"
        return Answer(
            kind="noul",
            prediction=pred,
            confidence=max(p, 1.0 - p),
            probabilities={"true": p, "false": 1.0 - p},
        )
    probs_raw = raw.get("probabilities")
    if not isinstance(probs_raw, dict):
        raise ParseError(f"missing probabilities for {q.name}")
    probs = {str(k): _f(v) for k, v in cast(dict[str, Any], probs_raw).items()}
    labels = q.labels()
    probs = {lab: probs.get(lab, 0.0) for lab in labels}
    total = sum(probs.values())
    if total <= 0:
        raise ParseError(f"degenerate probabilities for {q.name}")
    probs = {k: v / total for k, v in probs.items()}
    if q.kind == "choice":
        choice = str(raw.get("choice"))
        if choice not in probs:
            choice = max(probs, key=lambda k: probs[k])
        return Answer(
            kind="choice", prediction=choice, confidence=probs[choice], probabilities=probs
        )
    # score
    expected = sum(int(k) * v for k, v in probs.items())
    if isinstance(raw.get("score"), int | float):
        expected = _f(raw["score"])
    mode = max(probs, key=lambda k: probs[k])
    return Answer(
        kind="score",
        prediction=mode,
        confidence=probs[mode],
        probabilities=probs,
        expected_score=expected,
    )


class SystemOneJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    @property
    def endpoint(self) -> str:
        return f"{OPENROUTER_BASE}/systemone"

    def build_request(self, spec: TaskSpec, state: dict[str, str]) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "state": dict(state),
            "questions": {q.name: question_payload(q) for q in spec.questions},
        }

    def estimate_cost(self, body: dict[str, Any]) -> float:
        tokens = estimate_tokens(json.dumps(body)) + 300
        return tokens * self.config.price_in_per_m / 1e6

    def parse(self, spec: TaskSpec, body: dict[str, Any]) -> ParsedResponse:
        answers_raw = body.get("answers")
        if not isinstance(answers_raw, dict):
            raise ParseError("response has no 'answers' object")
        answers_d = cast(dict[str, Any], answers_raw)
        answers: dict[str, Answer] = {}
        for q in spec.questions:
            raw = answers_d.get(q.name)
            if not isinstance(raw, dict):
                raise ParseError(f"missing answer for {q.name}")
            answers[q.name] = parse_answer(q, cast(dict[str, Any], raw))
        usage = cast(dict[str, Any], body.get("usage") or {})
        return ParsedResponse(
            answers=answers,
            model_served=cast(str | None, body.get("model")),
            provider=cast(str | None, body.get("provider")),
            cost_usd=_opt_float(usage.get("cost")),
            input_tokens=_opt_int(usage.get("input_tokens")),
            output_tokens=_opt_int(usage.get("output_tokens")),
        )


def _opt_float(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and not isinstance(x, bool) else None


def _opt_int(x: Any) -> int | None:
    return int(x) if isinstance(x, int | float) and not isinstance(x, bool) else None
