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
