# [협업 주석]
# Goal: 모든 estimator filter가 따를 공통 BaseFilter interface를 정의한다.
# What it does: predict, update, get_state, reset, get_covariance API contract를 제공한다.
"""Base interface for state estimation filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseFilter(ABC):
    """Minimal benchmark-friendly filter interface."""

    @abstractmethod
    def predict(self, u: Optional[np.ndarray], dt: float) -> None:
        """Propagate filter state forward in time."""

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """Update filter state using a measurement."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current state estimate."""

    @abstractmethod
    def reset(
        self,
        initial_mean: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        particles: Optional[np.ndarray] = None,
    ) -> None:
        """Reset internal filter state."""

    def get_covariance(self) -> Optional[np.ndarray]:
        """Optionally return estimated covariance."""
        return None
