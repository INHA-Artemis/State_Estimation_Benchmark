from __future__ import annotations

from typing import Iterable

import numpy as np

from models.state_model import state_vector, zero_state
from utils.math_utils import fit_diag, fit_vector, wrap_angle


class PoseFilterMixin:
    def initialize(self, mean: Iterable[float] | None = None, cov_diag: Iterable[float] | None = None) -> None:
        if mean is None:
            mean_vec = zero_state(self.pose_type)
        else:
            mean_vec = state_vector(fit_vector(np.asarray(mean, dtype=float).reshape(-1), self.dim), self.pose_type)

        cov = fit_diag(np.zeros(self.dim) if cov_diag is None else cov_diag, self.dim)
        self.x = self._normalize_angles(mean_vec)
        self.P = np.diag(np.clip(cov, 1e-12, None)).astype(float)
        self.initialized = True

    def step(
        self,
        control: Iterable[float] | None,
        measurement: Iterable[float] | None,
        dt: float,
        mode: str | None = None,
    ) -> np.ndarray:
        run_mode = self.mode if mode is None else mode
        if run_mode in ("imu_only", "fused"):
            self.predict(control, dt)
        if run_mode in ("gnss_only", "fused"):
            self.measurement_update(measurement)
        return self.estimate_pose()

    def run(self, dataset: list[dict], mode: str | None = None) -> np.ndarray:
        estimates = []
        for sample in dataset:
            estimates.append(
                self.step(
                    sample.get("control"),
                    sample.get("measurement"),
                    float(sample.get("dt", 1.0)),
                    mode=mode,
                )
            )
        if not estimates:
            return np.zeros((0, self.dim), dtype=float)
        return np.vstack(estimates)

    def estimate_pose(self) -> np.ndarray:
        return self._normalize_angles(self.x.copy())

    def _transition_function(self, x: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        if self.pose_type == "2d":
            if control.size < 2:
                raise ValueError("2D control must contain at least [speed, yaw_rate].")
            speed, yaw_rate = control[0], control[1]
            yaw = x[2]
            x_next = x.copy()
            x_next[0] += speed * np.cos(yaw) * dt
            x_next[1] += speed * np.sin(yaw) * dt
            x_next[2] += yaw_rate * dt
            return self._normalize_angles(x_next)

        if control.size < 6:
            raise ValueError("3D control must contain at least 6 values.")
        x_next = x.copy()
        x_next[:6] += control[:6] * dt
        return self._normalize_angles(x_next)

    def _normalize_angles(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        for idx in self._angle_indices():
            x[idx] = wrap_angle(x[idx])
        return x

    def _angle_indices(self) -> list[int]:
        return [2] if self.pose_type == "2d" else [3, 4, 5]
