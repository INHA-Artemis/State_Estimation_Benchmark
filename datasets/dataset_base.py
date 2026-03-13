# [협업 주석]
# Goal: dataset 입력 형식을 통일해 filter/runner 코드와 decouple한다.
# What it does: DatasetStep, DatasetSequence, DatasetLoader interface를 정의하여
# IMU/GPS/GT state를 시간축으로 제공하는 표준 구조를 만든다.
"""Dataset interfaces for estimator runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class DatasetStep:
    """Single timestamped sample containing control/measurement data."""

    t: float
    dt: float
    imu: Optional[np.ndarray]
    gps: Optional[np.ndarray]
    gt_state: Optional[np.ndarray] = None


class DatasetSequence:
    """Container for one sequence consumed by filters."""

    def __init__(self, steps: list[DatasetStep], pose_type: str, state_dim: int) -> None:
        self.steps = steps
        self.pose_type = pose_type
        self.state_dim = state_dim

    def __iter__(self) -> Iterator[DatasetStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


class DatasetLoader(ABC):
    """Abstract dataset loader."""

    @abstractmethod
    def load_sequence(self) -> DatasetSequence:
        """Return one benchmark sequence."""
