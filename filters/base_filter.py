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
        """
        Goal:
            filter state를 다음 time step으로 propagate하는 공통 interface를 정의한다.
        Input:
            u는 optional control input이고, dt는 time delta이다.
        Output:
            없음. concrete filter가 내부 state를 갱신해야 한다.
        """

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """
        Goal:
            measurement를 사용한 state correction interface를 정의한다.
        Input:
            z는 measurement vector이다.
        Output:
            없음. concrete filter가 내부 state를 갱신해야 한다.
        """

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Goal:
            현재 state estimate를 읽는 공통 interface를 정의한다.
        Input:
            self는 concrete filter instance이다.
        Output:
            현재 state estimate numpy array를 반환해야 한다.
        """

    @abstractmethod
    def reset(
        self,
        initial_mean: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        particles: Optional[np.ndarray] = None,
    ) -> None:
        """
        Goal:
            filter internal state를 초기화하는 공통 interface를 정의한다.
        Input:
            initial_mean, initial_cov, particles는 filter 종류에 따라 사용할 초기화 정보이다.
        Output:
            없음. concrete filter가 내부 state를 초기 상태로 재설정해야 한다.
        """

    def get_covariance(self) -> Optional[np.ndarray]:
        """
        Goal:
            filter가 covariance 추정을 제공할 때 읽을 수 있는 기본 hook를 제공한다.
        Input:
            self는 concrete filter instance이다.
        Output:
            covariance를 지원하면 numpy array를, 아니면 None을 반환한다.
        """
        return None
