"""Async HTTP with bounded retries, exponential backoff and timeouts."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 8
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(self.max_delay_s, max(0.0, float(retry_after)))
            except ValueError:
                pass
        backoff = self.base_delay_s * (2 ** (attempt - 1))
        return min(self.max_delay_s, backoff) * (0.5 + random.random() / 2)


@dataclass(frozen=True)
class HttpResult:
    body: dict[str, Any]
    latency_s: float
    attempts: int


def _embedded_error_status(body: dict[str, Any]) -> int | None:
    err = body.get("error")
    if isinstance(err, dict):
        code = cast(dict[str, Any], err).get("code")
        if isinstance(code, int):
            return code
        return 500
    return None


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    policy: RetryPolicy | None = None,
) -> HttpResult:
    """POST JSON, retrying transport errors, 429 and 5xx (including errors that
    OpenRouter embeds in a 200 body). Latency is that of the successful attempt."""
    policy = policy or RetryPolicy()
    last: ProviderError | None = None
    for attempt in range(1, policy.max_attempts + 1):
        start = time.perf_counter()
        retry_after: str | None = None
        try:
            resp = await client.post(url, json=payload, headers=headers)
            latency = time.perf_counter() - start
            try:
                body: Any = resp.json()
            except ValueError:
                body = {"error": {"code": resp.status_code, "message": resp.text[:500]}}
            if not isinstance(body, dict):
                body = {"error": {"code": resp.status_code, "message": "non-object JSON"}}
            body_d = cast(dict[str, Any], body)
            status = resp.status_code if resp.status_code >= 400 else _embedded_error_status(body_d)
            if status is None:
                return HttpResult(body=body_d, latency_s=latency, attempts=attempt)
            last = ProviderError(f"HTTP {status}: {str(body_d.get('error'))[:300]}", status, body_d)
            retry_after = resp.headers.get("retry-after")
            if status not in RETRYABLE_STATUS:
                raise last
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last = ProviderError(f"transport error: {exc!r}")
        if attempt < policy.max_attempts:
            await asyncio.sleep(policy.delay(attempt, retry_after))
    assert last is not None
    raise last
