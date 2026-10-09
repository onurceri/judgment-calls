from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_json():
    def load(name: str) -> dict[str, Any]:
        return json.loads((FIXTURES / name).read_text())

    return load


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never touch the network."""
    import httpx

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("network access in tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", boom)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", boom)
