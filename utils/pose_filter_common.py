from __future__ import annotations

from typing import Iterable

import numpy as np

from models.state_model import state_vector, zero_state
from utils.math_utils import exp_so3, fit_diag, fit_vector, wrap_angle
from utils.rotation_utils import rot_to_rpy, rpy_to_rot


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
            return np.zeros((0, 3 if self.pose_type == "2d" else 6), dtype=float)
        return np.vstack(estimates)

    def estimate_pose(self) -> np.ndarray:
        x = self._normalize_angles(self.x.copy())
        if self.pose_type == "2d":
            return x
        if x.size >= 9:
            return np.concatenate([x[0:3], x[6:9]])
        return x[:6]

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

        # Legacy fallback for older 3D pose-only state [x, y, z, roll, pitch, yaw]
        if x_next.size < 9:
            x_next[:6] += control[:6] * dt
            return self._normalize_angles(x_next)

        linear = np.asarray(control[0:3], dtype=float)
        angular = np.asarray(control[3:6], dtype=float)

        p_prev = x_next[0:3]
        v_prev = x_next[3:6]
        rpy_prev = x_next[6:9]

        if x_next.size >= 15:
            b_a = x_next[9:12]
            b_w = x_next[12:15]
        else:
            b_a = np.zeros(3, dtype=float)
            b_w = np.zeros(3, dtype=float)

        linear_input_type = str(getattr(self, "linear_input_type", "velocity")).lower()
        gravity = np.asarray(getattr(self, "gravity", np.array([0.0, 0.0, -9.81], dtype=float)), dtype=float).reshape(-1)
        if gravity.size != 3:
            raise ValueError("gravity must have 3 values.")

        corrected_linear = linear - b_a
        corrected_angular = angular - b_w

        R_prev = rpy_to_rot(rpy_prev)
        R_t = R_prev @ exp_so3(corrected_angular * dt)

        if linear_input_type == "acceleration":
            accel_world = R_prev @ corrected_linear + gravity
            v_next = v_prev + accel_world * dt
            p_next = p_prev + v_prev * dt + 0.5 * accel_world * (dt**2)
        else:
            v_next = R_prev @ corrected_linear
            p_next = p_prev + v_next * dt

        x_next[0:3] = p_next
        x_next[3:6] = v_next
        x_next[6:9] = rot_to_rpy(R_t)
        return self._normalize_angles(x_next)

    def _normalize_angles(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        for idx in self._angle_indices():
            x[idx] = wrap_angle(x[idx])
        return x

    def _angle_indices(self) -> list[int]:
        if self.pose_type == "2d":
            return [2]
        if self.dim >= 9:
            return [6, 7, 8]
        return [3, 4, 5]
