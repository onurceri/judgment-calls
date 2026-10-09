"""Core data models shared across tasks, providers, the runner and the report."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionKind = Literal["noul", "choice", "score"]
JudgeKind = Literal["decision", "generative"]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QuestionSpec(BaseModel):
    """One judgment question. The same text feeds both judge families.

    criteria:
      * noul   -> {"true": ..., "false": ...}
      * choice -> {label: description}
      * score  -> ordered list of level descriptions (index 0 = lowest)
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: QuestionKind
    instructions: str
    criteria: dict[str, str] | list[str]

    def labels(self) -> list[str]:
        """Answer labels in canonical order (score levels are "0".."n-1")."""
        if self.kind == "score":
            assert isinstance(self.criteria, list)
            return [str(i) for i in range(len(self.criteria))]
        assert isinstance(self.criteria, dict)
        if self.kind == "noul":
            return ["true", "false"]
        return list(self.criteria.keys())


class TaskSpec(BaseModel):
    """A versioned judge task. Change `version` whenever any wording changes."""

    model_config = ConfigDict(frozen=True)

    task: str
    version: str
    description: str
    questions: list[QuestionSpec]

    def spec_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump()))[:16]

    def question(self, name: str) -> QuestionSpec:
        for q in self.questions:
            if q.name == name:
                return q
        raise KeyError(name)


class Item(BaseModel):
    """One human-labeled example.

    `state` is an ordered mapping of named text fields shown to the judge.
    `gold` maps question name -> human label (bool for noul, label str for choice,
    float mean rating on the level-index scale for score).
    """

    task: str
    item_id: str
    state: dict[str, str]
    gold: dict[str, bool | str | float]
    meta: dict[str, Any] = Field(default_factory=dict[str, Any])


class Variant(BaseModel):
    """A concrete presentation of an item (e.g. original vs. A/B-swapped order)."""

    name: str
    state: dict[str, str]
    gold: dict[str, bool | str | float]


class Answer(BaseModel):
    """A judge's parsed answer to one question.

    probabilities: distribution over QuestionSpec.labels() if the judge exposes one
      (decision models always; generative judges only via logprobs).
    confidence: probability-like confidence in `prediction` that is used for
      calibration. For decision models this is the model's own probability; for
      generative judges it is the verbalized confidence (0-100 rescaled to 0-1).
    """

    kind: QuestionKind
    prediction: str  # label from QuestionSpec.labels()
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    expected_score: float | None = None  # score questions: expected level index
    verbalized_confidence: float | None = None
    logprob_probabilities: dict[str, float] | None = None


class JudgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: JudgeKind
    model: str
    price_in_per_m: float
    price_out_per_m: float = 0.0
    context_tokens: int | None = None
    logprobs: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict[str, Any])
    tier: Literal["primary", "secondary", "reference"] = "primary"
    notes: str = ""


class JudgmentRecord(BaseModel):
    """One row of the append-only raw results log (full provenance)."""

    run_id: str
    timestamp: str
    task: str
    spec_version: str
    spec_hash: str
    item_id: str
    variant: str
    repeat: int
    judge_id: str
    judge_kind: JudgeKind
    model_requested: str
    model_served: str | None = None
    provider: str | None = None
    request_hash: str
    cached: bool = False
    latency_s: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    answers: dict[str, Answer] = Field(default_factory=dict[str, Answer])
    gold: dict[str, bool | str | float] = Field(default_factory=dict[str, bool | str | float])
    approx_tokens: int | None = None
    error: str | None = None
    raw_response: dict[str, Any] | None = None
