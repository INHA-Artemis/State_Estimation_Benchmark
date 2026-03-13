# [협업 주석]
# Goal: YAML config 로딩을 재사용 가능한 utility로 제공한다.
# What it does: 파일 존재/형식 검증 후 safe_load로 dict를 반환하고, 오류 상황을 명확히 전달한다.
"""Reusable YAML config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML file into a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for config loading. Install with `pip install pyyaml`.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return data
