from __future__ import annotations

import pytest

from judgment_calls.judges import JUDGES_BY_ID
from judgment_calls.providers.base import ParseError
from judgment_calls.providers.chat_judge import (
    ChatJudge,
    render_user_prompt,
    response_schema,
    value_distribution,
)
from judgment_calls.providers.systemone import SystemOneJudge
from judgment_calls.tasks import faithfulness, pairwise, summeval

STATE = {"response": "Paris is in Italy.", "source": "Paris is the capital of France."}


def test_systemone_request_uses_spec_wording() -> None:
    judge = SystemOneJudge(JUDGES_BY_ID["jev-1.13"])
    body = judge.build_request(faithfulness.SPEC, STATE)
    q = faithfulness.SPEC.questions[0]
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == STATE
    assert body["questions"]["hallucination"] == {
        "type": "noul",
        "instructions": q.instructions,
        "criteria": dict(q.criteria),
    }
    score = judge.build_request(summeval.SPEC, {"summary": "s", "article": "a"})
    assert set(score["questions"]) == {"coherence", "consistency", "fluency", "relevance"}
    assert score["questions"]["fluency"]["criteria"] == summeval.LEVELS


def test_chat_request_uses_same_wording_and_strict_schema() -> None:
    judge = ChatJudge(JUDGES_BY_ID["gpt-6-luna"])
    body = judge.build_request(
        pairwise.SPEC, {"user_message": "q", "answer_A": "a", "answer_B": "b"}
    )
    prompt = body["messages"][1]["content"]
    q = pairwise.SPEC.questions[0]
    assert q.instructions in prompt
    assert isinstance(q.criteria, dict)
    for desc in q.criteria.values():
        assert desc in prompt
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.0
    assert "logprobs" not in body
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["preference"] == {"type": "string", "enum": ["A", "B", "tie"]}
    assert schema["required"] == ["preference", "preference_confidence"]
    assert schema["additionalProperties"] is False
    lp = ChatJudge(JUDGES_BY_ID["gpt-4o-mini"]).build_request(pairwise.SPEC, {"x": "y"})
    assert lp["logprobs"] is True
    assert lp["top_logprobs"] == 10


def test_score_schema_and_prompt_levels() -> None:
    schema = response_schema(summeval.SPEC)
    assert schema["properties"]["coherence"] == {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    prompt = render_user_prompt(summeval.SPEC, {"summary": "s", "article": "a"})
    for level in summeval.LEVELS:
        assert level in prompt
    assert prompt.index("<summary>") < prompt.index("<article>")


def test_parse_systemone_fixtures(fixture_json) -> None:
    judge = SystemOneJudge(JUDGES_BY_ID["jev-1.13"])
    noul = judge.parse(faithfulness.SPEC, fixture_json("systemone_noul.json"))
    a = noul.answers["hallucination"]
    p = a.probabilities
    assert p is not None
    assert p["true"] + p["false"] == pytest.approx(1.0)
    assert a.prediction == ("true" if p["true"] >= 0.5 else "false")
    assert a.confidence == pytest.approx(max(p.values()))
    assert noul.cost_usd is not None
    assert noul.model_served

    score = judge.parse(summeval.SPEC, fixture_json("systemone_score.json"))
    for name in ("coherence", "consistency", "fluency", "relevance"):
        s = score.answers[name]
        assert s.probabilities is not None
        assert set(s.probabilities) == {"0", "1", "2", "3", "4"}
        assert s.expected_score is not None
        assert 0.0 <= s.expected_score <= 4.0
        manual = sum(int(k) * v for k, v in s.probabilities.items())
        assert s.expected_score == pytest.approx(manual, abs=0.05)

    choice = judge.parse(pairwise.SPEC, fixture_json("systemone_choice.json"))
    c = choice.answers["preference"]
    assert c.prediction == "B"
    assert c.probabilities is not None
    assert c.confidence == pytest.approx(c.probabilities["B"])


def test_parse_systemone_rejects_missing_answers() -> None:
    judge = SystemOneJudge(JUDGES_BY_ID["jev-1.13"])
    with pytest.raises(ParseError):
        judge.parse(faithfulness.SPEC, {"answers": {}})
    with pytest.raises(ParseError):
        judge.parse(faithfulness.SPEC, {"answers": {"hallucination": {"noul": 1.7}}})


def test_parse_chat_with_logprobs(fixture_json) -> None:
    judge = ChatJudge(JUDGES_BY_ID["gpt-4o-mini"])
    r = judge.parse(pairwise.SPEC, fixture_json("chat_logprobs_choice.json"))
    a = r.answers["preference"]
    assert a.prediction == "B"
    assert a.verbalized_confidence == pytest.approx(0.9)
    assert a.logprob_probabilities is not None
    assert sum(a.logprob_probabilities.values()) == pytest.approx(1.0)
    assert max(a.logprob_probabilities, key=a.logprob_probabilities.__getitem__) == "B"

    noul = judge.parse(faithfulness.SPEC, fixture_json("chat_logprobs_noul.json"))
    lp = noul.answers["hallucination"].logprob_probabilities
    assert lp is not None
    assert set(lp) == {"true", "false"}

    score = judge.parse(summeval.SPEC, fixture_json("chat_logprobs_score.json"))
    for name in ("coherence", "consistency", "fluency", "relevance"):
        s = score.answers[name]
        assert s.logprob_probabilities is not None
        assert set(s.logprob_probabilities) == {"0", "1", "2", "3", "4"}
        assert s.expected_score is not None
        assert abs(s.expected_score - int(s.prediction)) < 1.0


def test_parse_chat_without_logprobs(fixture_json) -> None:
    judge = ChatJudge(JUDGES_BY_ID["claude-haiku-5.5"])
    r = judge.parse(pairwise.SPEC, fixture_json("chat_plain_choice.json"))
    a = r.answers["preference"]
    assert a.prediction in {"A", "B", "tie"}
    assert a.logprob_probabilities is None
    assert a.confidence is not None
    assert 0.0 <= a.confidence <= 1.0
    assert r.cost_usd is not None


def test_parse_chat_errors() -> None:
    judge = ChatJudge(JUDGES_BY_ID["claude-haiku-5.5"])
    empty = {"choices": [{"message": {"content": None}}]}
    with pytest.raises(ParseError):
        judge.parse(pairwise.SPEC, empty)
    bad = {"choices": [{"message": {"content": '{"preference": "C", "preference_confidence": 5}'}}]}
    with pytest.raises(ParseError):
        judge.parse(pairwise.SPEC, bad)
    fenced = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"preference": "tie", "preference_confidence": 55}\n```'
                }
            }
        ]
    }
    assert judge.parse(pairwise.SPEC, fenced).answers["preference"].prediction == "tie"


def test_value_distribution_with_quote_prefixed_tokens() -> None:
    content = '{"preference":"tie","preference_confidence":60}'
    tokens = [
        {"token": '{"preference":', "logprob": 0.0},
        {
            "token": '"tie',
            "logprob": -0.5,
            "top_logprobs": [
                {"token": '"tie', "logprob": -0.5},
                {"token": '"A', "logprob": -1.0},
                {"token": "A", "logprob": -2.0},  # different tokenization: ignored
            ],
        },
        {"token": '","preference_confidence":60}', "logprob": 0.0},
    ]
    dist = value_distribution(tokens, content, "preference", ["A", "B", "tie"])
    assert dist is not None
    import math

    z = math.exp(-0.5) + math.exp(-1.0)
    assert dist["tie"] == pytest.approx(math.exp(-0.5) / z)
    assert dist["A"] == pytest.approx(math.exp(-1.0) / z)
    assert dist["B"] == 0.0
    assert value_distribution(tokens, content, "missing", ["A"]) is None
