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
    """
    Goal:
        한 번의 benchmark 실행 결과를 저장 가능한 summary payload로 묶는다.
    Input:
        filter_name, mode, dataset_type, pose_type은 실행 metadata이고, metrics는 계산된 metric dict이다.
    Output:
        JSON serialization 가능한 summary dict를 반환한다.
    """
    return {
        "filter": filter_name,
        "mode": mode,
        "dataset_type": dataset_type,
        "pose_type": pose_type,
        "metrics": metrics,
    }


def save_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    """
    Goal:
        summary dict를 JSON file로 저장한다.
    Input:
        summary는 저장할 결과 dict이고, output_path는 생성할 JSON 경로이다.
    Output:
        실제로 저장된 JSON file의 Path를 반환한다.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
