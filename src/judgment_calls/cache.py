"""Content-addressed response cache (SQLite) so re-runs never pay twice."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from judgment_calls.models import canonical_json, sha256_hex


def request_hash(endpoint: str, body: dict[str, Any], repeat: int) -> str:
    """Hash of everything that determines a response. `repeat` is part of the key so
    that deliberate repeats are real, independent calls."""
    return sha256_hex(canonical_json({"endpoint": endpoint, "body": body, "repeat": repeat}))


@dataclass(frozen=True)
class CachedResponse:
    body: dict[str, Any]
    latency_s: float
    created_at: str


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, body TEXT NOT NULL, latency_s REAL NOT NULL,"
            " created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> CachedResponse | None:
        row = self._conn.execute(
            "SELECT body, latency_s, created_at FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        body, latency, created = cast(tuple[str, float, str], row)
        return CachedResponse(json.loads(body), float(latency), created)

    def put(self, key: str, body: dict[str, Any], latency_s: float, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, body, latency_s, created_at) VALUES (?,?,?,?)",
            (key, json.dumps(body), latency_s, created_at),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
