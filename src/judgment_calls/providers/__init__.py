"""Judge providers: OpenRouter decision models (SystemOne) and generative chat judges."""

from __future__ import annotations

from judgment_calls.models import JudgeConfig
from judgment_calls.providers.base import Judge, ParsedResponse
from judgment_calls.providers.chat_judge import ChatJudge
from judgment_calls.providers.systemone import SystemOneJudge


def make_judge(config: JudgeConfig) -> Judge:
    if config.kind == "decision":
        return SystemOneJudge(config)
    return ChatJudge(config)


__all__ = ["ChatJudge", "Judge", "ParsedResponse", "SystemOneJudge", "make_judge"]
