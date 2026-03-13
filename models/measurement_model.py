# [협업 주석]
# Goal: filter와 분리된 MeasurementModel abstraction을 제공한다.
# What it does: likelihood interface와 Gaussian PositionMeasurementModel을 제공하여
# measurement z에 대한 p(z|x) / log-likelihood를 계산한다.
"""Measurement model interfaces and minimal implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class MeasurementModel(ABC):
    """Abstract measurement model used by filters."""

    @abstractmethod
    def likelihood(self, z: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Goal:
            measurement likelihood 계산 interface를 정의한다.
        Input:
            z는 measurement vector이고, states는 candidate state sample 배열이다.
        Output:
            각 state에 대한 likelihood numpy array를 반환해야 한다.
        """


class PositionMeasurementModel(MeasurementModel):
    """Gaussian position measurement model (e.g., GPS x-y)."""

    def __init__(self, position_indices: Sequence[int], measurement_noise_cov: np.ndarray) -> None:
        """
        Goal:
            Gaussian position MeasurementModel에 필요한 noise parameter를 준비한다.
        Input:
            position_indices는 관측할 state index들이고, measurement_noise_cov는 measurement covariance matrix이다.
        Output:
            없음. inverse covariance와 normalization constant가 미리 계산된다.
        """
        self.position_indices = tuple(int(i) for i in position_indices)
        self.R = np.asarray(measurement_noise_cov, dtype=float)
        self.dim = len(self.position_indices)

        if self.R.shape != (self.dim, self.dim):
            raise ValueError(f"measurement_noise_cov must be {(self.dim, self.dim)}, got {self.R.shape}")
        self.R_inv = np.linalg.inv(self.R)
        sign, logdet = np.linalg.slogdet(self.R)
        if sign <= 0:
            raise ValueError("measurement noise covariance must be positive definite")
        self.log_norm = -0.5 * (self.dim * np.log(2.0 * np.pi) + logdet)

    def likelihood(self, z: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Goal:
            position measurement에 대한 likelihood 값을 계산한다.
        Input:
            z는 관측 position vector이고, states는 candidate state sample 배열이다.
        Output:
            각 state sample에 대한 likelihood numpy array를 반환한다.
        """
        return np.exp(self.log_likelihood(z, states))

    def log_likelihood(self, z: np.ndarray, states: np.ndarray) -> np.ndarray:
        """
        Goal:
            position measurement에 대한 Gaussian log-likelihood를 계산한다.
        Input:
            z는 관측 position vector이고, states는 candidate state sample 배열이다.
        Output:
            각 state sample에 대한 log-likelihood numpy array를 반환한다.
        """
        z = np.asarray(z, dtype=float).reshape(self.dim)
        predicted = states[:, self.position_indices]
        residual = predicted - z[None, :]
        maha = np.einsum("ni,ij,nj->n", residual, self.R_inv, residual)
        return self.log_norm - 0.5 * maha
