"""Generative LLM-as-a-judge via OpenRouter chat completions with structured output.

The prompt and JSON schema are generated from the same :class:`TaskSpec` that feeds
the decision models, so both judge families see identical instructions and criteria.
For every question the judge returns its answer plus a verbalized confidence (0-100).
If the judge exposes token logprobs, the probability distribution over the answer
token is extracted as a second, non-verbal confidence signal.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, cast

from judgment_calls.models import Answer, JudgeConfig, QuestionSpec, TaskSpec
from judgment_calls.providers.base import (
    OPENROUTER_BASE,
    ParsedResponse,
    ParseError,
    estimate_tokens,
)

SYSTEM_PROMPT = (
    "You are an impartial, careful evaluator. Read the input fields, then answer every "
    "question exactly as defined, using only the information given. For each answer also "
    "state how confident you are that it is correct, as an integer from 0 to 100. "
    "Reply with JSON that matches the provided schema and nothing else."
)
MAX_OUTPUT_TOKENS = 200


def confidence_field(q: QuestionSpec) -> str:
    return f"{q.name}_confidence"


def render_state(state: dict[str, str]) -> str:
    return "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in state.items())


def render_question(q: QuestionSpec) -> str:
    lines = [f'### Question "{q.name}"', q.instructions]
    if q.kind == "noul":
        assert isinstance(q.criteria, dict)
        lines.append(f"Answer true if: {q.criteria['true']}")
        lines.append(f"Answer false if: {q.criteria['false']}")
    elif q.kind == "choice":
        assert isinstance(q.criteria, dict)
        lines.append("Answer with exactly one of these options:")
        lines.extend(f'- "{label}": {desc}' for label, desc in q.criteria.items())
    else:
        assert isinstance(q.criteria, list)
        n = len(q.criteria)
        lines.append(
            f"Answer with an integer from 1 to {n}: the position of the matching level in "
            "this list, lowest first:"
        )
        lines.extend(f"{i + 1}. {level}" for i, level in enumerate(q.criteria))
    lines.append(
        f'Put your answer in "{q.name}" and your confidence (0-100) that it is correct in '
        f'"{confidence_field(q)}".'
    )
    return "\n".join(lines)


def render_user_prompt(spec: TaskSpec, state: dict[str, str]) -> str:
    parts = ["## Input", render_state(state), "## Questions"]
    parts.extend(render_question(q) for q in spec.questions)
    return "\n\n".join(parts)


def answer_schema(q: QuestionSpec) -> dict[str, Any]:
    if q.kind == "noul":
        return {"type": "boolean"}
    if q.kind == "choice":
        return {"type": "string", "enum": q.labels()}
    assert isinstance(q.criteria, list)
    return {"type": "integer", "enum": list(range(1, len(q.criteria) + 1))}


def response_schema(spec: TaskSpec) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for q in spec.questions:
        props[q.name] = answer_schema(q)
        props[confidence_field(q)] = {"type": "integer"}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# logprob extraction


def _token_text(tok: dict[str, Any]) -> str:
    return str(tok.get("token", ""))


def value_distribution(
    logprob_tokens: list[dict[str, Any]], content: str, field: str, labels: list[str]
) -> dict[str, float] | None:
    """Probability distribution over `labels` at the position where the JSON value of
    `field` starts, using the token's top-k alternatives. Returns None if the value
    token cannot be located or no alternative maps to a label."""
    m = re.search(r'"' + re.escape(field) + r'"\s*:\s*', content)
    if m is None:
        return None
    value_start = m.end()
    if value_start < len(content) and content[value_start] == '"':
        value_start += 1
    offset = 0
    for tok in logprob_tokens:
        text = _token_text(tok)
        start, end = offset, offset + len(text)
        offset = end
        if not start <= value_start < end:
            continue
        prefix = content[start:value_start]  # e.g. '"' when the token is '"A'
        mass: dict[str, float] = {}
        alts = cast(list[dict[str, Any]], tok.get("top_logprobs") or [tok])
        for alt in alts:
            alt_text = _token_text(alt)
            if not alt_text.lstrip().startswith(prefix.lstrip()):
                continue
            rest = alt_text.lstrip()[len(prefix.lstrip()) :].strip().strip('"')
            if not rest:
                continue
            matches = [lab for lab in labels if lab.startswith(rest) or rest.startswith(lab)]
            if len(matches) != 1:
                continue
            lp = alt.get("logprob")
            if not isinstance(lp, int | float):
                continue
            mass[matches[0]] = mass.get(matches[0], 0.0) + math.exp(float(lp))
        total = sum(mass.values())
        if total <= 0:
            return None
        return {lab: mass.get(lab, 0.0) / total for lab in labels}
    return None


# ---------------------------------------------------------------------------


def _parse_confidence(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return min(100.0, max(0.0, float(x))) / 100.0


def parse_answer(
    q: QuestionSpec,
    obj: dict[str, Any],
    logprob_tokens: list[dict[str, Any]] | None,
    content: str,
) -> Answer:
    if q.name not in obj:
        raise ParseError(f"missing field {q.name}")
    value = obj[q.name]
    verbal = _parse_confidence(obj.get(confidence_field(q)))
    labels = q.labels()
    if q.kind == "noul":
        if not isinstance(value, bool):
            raise ParseError(f"{q.name}: expected boolean, got {value!r}")
        pred = "true" if value else "false"
        lp_labels = labels
    elif q.kind == "choice":
        pred = str(value)
        if pred not in labels:
            raise ParseError(f"{q.name}: {pred!r} not in {labels}")
        lp_labels = labels
    else:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ParseError(f"{q.name}: expected integer, got {value!r}")
        level = int(value) - 1
        if not 0 <= level < len(labels):
            raise ParseError(f"{q.name}: score {value} out of range")
        pred = str(level)
        lp_labels = [str(i + 1) for i in range(len(labels))]  # tokens are 1-based digits

    lp_dist: dict[str, float] | None = None
    if logprob_tokens:
        dist = value_distribution(logprob_tokens, content, q.name, lp_labels)
        if dist is not None:
            lp_dist = {str(int(k) - 1): v for k, v in dist.items()} if q.kind == "score" else dist

    expected: float | None = None
    if q.kind == "score":
        expected = (
            sum(int(k) * v for k, v in lp_dist.items()) if lp_dist is not None else float(pred)
        )
    return Answer(
        kind=q.kind,
        prediction=pred,
        confidence=verbal,
        verbalized_confidence=verbal,
        logprob_probabilities=lp_dist,
        expected_score=expected,
    )


class ChatJudge:
    def __init__(self, config: JudgeConfig, temperature: float = 0.0) -> None:
        self.config = config
        self.temperature = temperature

    @property
    def endpoint(self) -> str:
        return f"{OPENROUTER_BASE}/chat/completions"

    def build_request(self, spec: TaskSpec, state: dict[str, str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_prompt(spec, state)},
            ],
            "temperature": self.temperature,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{spec.task}_judgment",
                    "strict": True,
                    "schema": response_schema(spec),
                },
            },
            "usage": {"include": True},
        }
        if self.config.logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 10
        body.update(self.config.extra_body)
        return body

    def estimate_cost(self, body: dict[str, Any]) -> float:
        tokens_in = estimate_tokens(json.dumps(body["messages"])) + 300
        return (
            tokens_in * self.config.price_in_per_m + MAX_OUTPUT_TOKENS * self.config.price_out_per_m
        ) / 1e6

    def parse(self, spec: TaskSpec, body: dict[str, Any]) -> ParsedResponse:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ParseError("no choices in response")
        choice = cast(dict[str, Any], cast(list[Any], choices)[0])
        message = cast(dict[str, Any], choice.get("message") or {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ParseError("no message content")
        obj = _loads_json_object(content)
        lp = cast(dict[str, Any] | None, choice.get("logprobs"))
        tokens = cast(list[dict[str, Any]] | None, lp.get("content") if lp else None)
        answers = {q.name: parse_answer(q, obj, tokens, content) for q in spec.questions}
        usage = cast(dict[str, Any], body.get("usage") or {})
        cost = usage.get("cost")
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        return ParsedResponse(
            answers=answers,
            model_served=cast(str | None, body.get("model")),
            provider=cast(str | None, body.get("provider")),
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
            input_tokens=int(pt) if isinstance(pt, int) else None,
            output_tokens=int(ct) if isinstance(ct, int) else None,
        )


def _loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m is None:
            raise ParseError(f"no JSON object in content: {content[:200]!r}") from None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    return cast(dict[str, Any], obj)
