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
        """
        Goal:
            sequence metadata와 step 목록을 container에 저장한다.
        Input:
            steps는 DatasetStep list이고, pose_type은 pose 표현 문자열이며, state_dim은 state 크기이다.
        Output:
            없음. DatasetSequence instance field를 초기화한다.
        """
        self.steps = steps
        self.pose_type = pose_type
        self.state_dim = state_dim

    def __iter__(self) -> Iterator[DatasetStep]:
        """
        Goal:
            DatasetSequence를 iterator처럼 순회 가능하게 만든다.
        Input:
            self는 DatasetStep 목록을 가진 DatasetSequence instance이다.
        Output:
            내부 steps를 순회하는 iterator를 반환한다.
        """
        return iter(self.steps)

    def __len__(self) -> int:
        """
        Goal:
            sequence 길이를 빠르게 조회할 수 있게 한다.
        Input:
            self는 DatasetStep 목록을 가진 DatasetSequence instance이다.
        Output:
            step 개수를 int로 반환한다.
        """
        return len(self.steps)


class DatasetLoader(ABC):
    """Abstract dataset loader."""

    @abstractmethod
    def load_sequence(self) -> DatasetSequence:
        """
        Goal:
            dataset loader가 공통 형식의 sequence를 제공하도록 강제한다.
        Input:
            self는 concrete DatasetLoader implementation instance이다.
        Output:
            benchmark가 바로 소비할 수 있는 DatasetSequence를 반환해야 한다.
        """
