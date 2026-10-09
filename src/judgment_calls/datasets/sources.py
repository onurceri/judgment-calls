"""Raw dataset sources (URLs, licenses) and download helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class RawFile:
    name: str
    url: str


RAW_FILES: dict[str, list[RawFile]] = {
    "faithfulness": [
        RawFile(
            "ragtruth_response.jsonl",
            "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
        ),
        RawFile(
            "ragtruth_source_info.jsonl",
            "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
        ),
    ],
    "summeval": [
        RawFile(
            "summeval.parquet",
            "https://huggingface.co/datasets/mteb/summeval/resolve/main/data/"
            "test-00000-of-00001-35901af5f6649399.parquet",
        )
    ],
    "pairwise": [
        RawFile(
            "mtbench_human.parquet",
            "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/"
            "human-00000-of-00001-25f4910818759289.parquet",
        )
    ],
}


def download(task: str, raw_dir: Path, *, force: bool = False) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for f in RAW_FILES[task]:
            dest = raw_dir / f.name
            if force or not dest.exists():
                resp = client.get(f.url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            paths.append(dest)
    return paths
