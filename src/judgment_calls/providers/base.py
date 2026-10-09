"""Provider interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from judgment_calls.models import Answer, JudgeConfig, TaskSpec

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class ParseError(ValueError):
    pass


@dataclass
class ParsedResponse:
    answers: dict[str, Answer]
    model_served: str | None = None
    provider: str | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


class Judge(Protocol):
    config: JudgeConfig

    @property
    def endpoint(self) -> str: ...

    def build_request(self, spec: TaskSpec, state: dict[str, str]) -> dict[str, Any]: ...

    def parse(self, spec: TaskSpec, body: dict[str, Any]) -> ParsedResponse: ...

    def estimate_cost(self, body: dict[str, Any]) -> float: ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)  # deliberately pessimistic (~3 chars/token)
