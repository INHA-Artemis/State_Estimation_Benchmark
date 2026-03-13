# [협업 주석]
# Goal: dataset_type 기반 loader 선택 로직을 중앙화한다.
# What it does: config 값을 해석해 해당 DatasetLoader를 생성하고, 미지원 타입은 명확히 에러 처리한다.
"""Dataset loader factory."""

from __future__ import annotations

from typing import Any

from datasets.dataset_base import DatasetLoader
from datasets.dummy_loader import DummySequenceLoader


def build_dataset_loader(cfg: dict[str, Any]) -> DatasetLoader:
    """Create dataset loader from config."""
    dataset_type = str(cfg.get("dataset_type", "dummy")).lower()

    if dataset_type == "dummy":
        return DummySequenceLoader(cfg)

    raise ValueError(
        f"Unsupported dataset_type='{dataset_type}'. "
        "Implement a loader and register it in datasets/factory.py."
    )
