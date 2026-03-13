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
        """Return measurement likelihood p(z|x) for each state sample."""


class PositionMeasurementModel(MeasurementModel):
    """Gaussian position measurement model (e.g., GPS x-y)."""

    def __init__(self, position_indices: Sequence[int], measurement_noise_cov: np.ndarray) -> None:
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
        return np.exp(self.log_likelihood(z, states))

    def log_likelihood(self, z: np.ndarray, states: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(self.dim)
        predicted = states[:, self.position_indices]
        residual = predicted - z[None, :]
        maha = np.einsum("ni,ij,nj->n", residual, self.R_inv, residual)
        return self.log_norm - 0.5 * maha
