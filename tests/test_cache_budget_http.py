from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from judgment_calls.budget import Budget, BudgetExceeded
from judgment_calls.cache import ResponseCache, request_hash
from judgment_calls.http import ProviderError, RetryPolicy, post_json


def test_request_hash_is_canonical_and_repeat_sensitive() -> None:
    a = request_hash("u", {"b": 1, "a": [1, 2]}, 0)
    b = request_hash("u", {"a": [1, 2], "b": 1}, 0)
    assert a == b
    assert request_hash("u", {"a": [1, 2], "b": 1}, 1) != a
    assert request_hash("v", {"a": [1, 2], "b": 1}, 0) != a


def test_cache_roundtrip_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    cache = ResponseCache(path)
    assert cache.get("k") is None
    cache.put("k", {"x": 1}, 0.25, "2026-01-01T00:00:00")
    cache.close()
    cache2 = ResponseCache(path)
    hit = cache2.get("k")
    assert hit is not None
    assert hit.body == {"x": 1}
    assert hit.latency_s == 0.25
    assert len(cache2) == 1


def test_budget_reserve_settle_and_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    b = Budget(0.010, ledger)
    b.reserve(0.004)
    b.settle(0.004, 0.003, "j")
    assert b.spent_usd == pytest.approx(0.003)
    b.reserve(0.006)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.002)  # 0.003 spent + 0.006 reserved + 0.002 > 0.010
    b.release(0.006)
    b.reserve(0.002)
    b.settle(0.002, None, "j")  # unknown actual cost -> books the estimate
    # ledger persists across instances
    assert Budget(0.010, ledger).spent_usd == pytest.approx(0.005)


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses.pop(0)


FAST = RetryPolicy(max_attempts=4, base_delay_s=0.0, max_delay_s=0.0)


async def _post(responses: list[httpx.Response]) -> tuple[Any, int]:
    transport = _Transport(responses)
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            res = await post_json(client, "https://x/api", {"a": 1}, {}, FAST)
            return res.body, transport.calls
        except ProviderError as exc:
            return exc, transport.calls


async def test_retries_429_and_5xx_then_succeeds() -> None:
    body, calls = await _post(
        [
            httpx.Response(429, json={"error": {"code": 429}}, headers={"retry-after": "0"}),
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    assert body == {"ok": True}
    assert calls == 3


async def test_retries_error_embedded_in_200_body() -> None:
    body, calls = await _post(
        [
            httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    assert body == {"ok": 1}
    assert calls == 2


async def test_does_not_retry_400() -> None:
    err, calls = await _post([httpx.Response(400, json={"error": {"code": 400}})])
    assert isinstance(err, ProviderError)
    assert err.status == 400
    assert calls == 1


async def test_gives_up_after_max_attempts() -> None:
    err, calls = await _post([httpx.Response(500, json={}) for _ in range(4)])
    assert isinstance(err, ProviderError)
    assert calls == 4
