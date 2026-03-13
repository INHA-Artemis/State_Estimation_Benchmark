# [협업 주석]
# Goal: filter와 분리된 MotionModel abstraction을 제공한다.
# What it does: propagate interface와 PlanarVelocityYawRateModel 구현을 통해
# control input(u=[v, yaw_rate]) 기반 state propagation(+process noise)을 수행한다.
"""Motion model interfaces and minimal implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from utils.math_utils import wrap_angle


class MotionModel(ABC):
    """Abstract motion model used by filters."""

    @abstractmethod
    def propagate(
        self,
        states: np.ndarray,
        u: Optional[np.ndarray],
        dt: float,
        noise: bool = True,
    ) -> np.ndarray:
        """Propagate one or more states forward."""


class PlanarVelocityYawRateModel(MotionModel):
    """
    Planar motion model using control u = [v, yaw_rate].

    Works for generic state vectors using configured indices.
    """

    def __init__(
        self,
        state_dim: int,
        process_noise_cov: np.ndarray,
        x_index: int = 0,
        y_index: int = 1,
        yaw_index: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.Q = np.asarray(process_noise_cov, dtype=float)
        self.x_index = int(x_index)
        self.y_index = int(y_index)
        self.yaw_index = int(yaw_index)
        self.rng = np.random.default_rng() if rng is None else rng

        if self.Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"process_noise_cov must be {(self.state_dim, self.state_dim)}, got {self.Q.shape}")

    def propagate(
        self,
        states: np.ndarray,
        u: Optional[np.ndarray],
        dt: float,
        noise: bool = True,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        out = states.copy()

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"states must be (N, {self.state_dim}), got {states.shape}")

        if u is None:
            v, yaw_rate = 0.0, 0.0
        else:
            u = np.asarray(u, dtype=float).reshape(-1)
            if u.size < 2:
                raise ValueError("u must contain at least [v, yaw_rate]")
            v, yaw_rate = float(u[0]), float(u[1])

        yaw = out[:, self.yaw_index]
        out[:, self.x_index] += v * dt * np.cos(yaw)
        out[:, self.y_index] += v * dt * np.sin(yaw)
        out[:, self.yaw_index] = wrap_angle(yaw + yaw_rate * dt)

        if noise:
            process_noise = self.rng.multivariate_normal(
                mean=np.zeros(self.state_dim),
                cov=self.Q,
                size=out.shape[0],
            )
            out += process_noise
            out[:, self.yaw_index] = wrap_angle(out[:, self.yaw_index])

        return out
