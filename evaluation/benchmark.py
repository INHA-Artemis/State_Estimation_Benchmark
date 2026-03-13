# [협업 주석]
# Goal: 다중 filter benchmark 실행/집계 로직을 담당한다.
# What it does: 각 filter 실행 결과를 수집하고 공통 비교 실험 흐름을 구성할 예정이다.
"""Benchmark result packaging and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_run_summary(
    filter_name: str,
    mode: str,
    dataset_type: str,
    pose_type: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build top-level result payload for a single estimator run."""
    return {
        "filter": filter_name,
        "mode": mode,
        "dataset_type": dataset_type,
        "pose_type": pose_type,
        "metrics": metrics,
    }


def save_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Save summary dictionary into a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
