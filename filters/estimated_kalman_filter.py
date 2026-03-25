from __future__ import annotations

from typing import Iterable

import numpy as np

from models import measurement_model
from models.state_model import state_dim, state_vector, zero_state


class ExtendedKalmanFilter:
    def __init__(
        self,
        pose_type: str = "2d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type not in ("2d", "3d"):
            raise ValueError("pose_type must be '2d' or '3d'.")

        self.pose_type = pose_type
        self.mode = mode
        self.dim = state_dim(pose_type)

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}

        default_meas_dim = 2 if pose_type == "2d" else 3
        self.process_noise_diag = self._fit_diag(motion_cfg.get("process_noise_diag", np.zeros(self.dim)), self.dim)
        self.measurement_indices = np.asarray(
            meas_cfg.get("position_indices", list(range(default_meas_dim))),
            dtype=int,
        )
        self.measurement_noise_diag = self._fit_diag(
            meas_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
        )

        self.x = zero_state(self.pose_type)
        self.P = np.eye(self.dim, dtype=float)
        self.initialized = False

    @classmethod
    def from_configs(cls, dataset_config: dict, ekf_config: dict) -> "ExtendedKalmanFilter":
        ekf = cls(
            pose_type=dataset_config.get("pose_type", "2d"),
            mode=dataset_config.get("mode", "fused"),
            motion_config=ekf_config.get("motion_model", {}),
            measurement_config=ekf_config.get("measurement_model", {}),
        )
        init_cfg = ekf_config.get("initialization", {})
        ekf.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"))
        return ekf

    def initialize(self, mean: Iterable[float] | None = None, cov_diag: Iterable[float] | None = None) -> None:
        if mean is None:
            mean_vec = zero_state(self.pose_type)
        else:
            mean_vec = state_vector(self._fit_vector(np.asarray(mean, dtype=float).reshape(-1), self.dim), self.pose_type)

        cov = self._fit_diag(np.zeros(self.dim) if cov_diag is None else cov_diag, self.dim)
        self.x = self._normalize_angles(mean_vec)
        self.P = np.diag(np.clip(cov, 1e-12, None)).astype(float)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()

        if control is None:
            x_pred = self.x.copy()
            F = np.eye(self.dim, dtype=float)
        else:
            u = np.asarray(control, dtype=float).reshape(-1)
            x_pred = self._transition_function(self.x, u, dt)
            F = self._transition_jacobian(self.x, u, dt)

        Q = np.diag(np.clip(self.process_noise_diag, 1e-12, None))
        self.x = self._normalize_angles(x_pred)
        self.P = self._nearest_spd(F @ self.P @ F.T + Q)
        return self.x.copy()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.x.copy()

        z = np.asarray(measurement, dtype=float).reshape(-1)
        if z.size != self.measurement_indices.size:
            raise ValueError("measurement size must match measurement indices.")

        z_pred = measurement_model.h(self.x, indices=self.measurement_indices)
        H = np.zeros((z.size, self.dim), dtype=float)
        for row, idx in enumerate(self.measurement_indices):
            H[row, idx] = 1.0

        R = np.diag(np.clip(self.measurement_noise_diag, 1e-12, None))
        innovation = z - z_pred
        S = self._nearest_spd(H @ self.P @ H.T + R)
        K = self.P @ H.T @ np.linalg.inv(S)

        I = np.eye(self.dim, dtype=float)
        self.x = self._normalize_angles(self.x + K @ innovation)
        self.P = self._nearest_spd((I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T)
        return self.x.copy()

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
        if run_mode in ("gps_only", "gnss_only", "fused"):
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

    def _transition_jacobian(self, x: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(self.dim, dtype=float)
        if self.pose_type == "2d":
            speed = float(control[0])
            yaw = float(x[2])
            F[0, 2] = -speed * np.sin(yaw) * dt
            F[1, 2] = speed * np.cos(yaw) * dt
        return F

    def _normalize_angles(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        for idx in self._angle_indices():
            x[idx] = self._wrap_angle(x[idx])
        return x

    def _angle_indices(self) -> list[int]:
        return [2] if self.pose_type == "2d" else [3, 4, 5]

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _nearest_spd(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, eps)
        spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return 0.5 * (spd + spd.T)

    @staticmethod
    def _fit_vector(values: np.ndarray, dim: int) -> np.ndarray:
        if values.size == dim:
            return values
        out = np.zeros(dim, dtype=float)
        out[: min(dim, values.size)] = values[: min(dim, values.size)]
        return out

    @staticmethod
    def _fit_diag(values: Iterable[float], dim: int) -> np.ndarray:
        diag = np.asarray(values, dtype=float).reshape(-1)
        if diag.size == dim:
            return diag
        if diag.size == 1:
            return np.full(dim, float(diag.item()), dtype=float)
        out = np.zeros(dim, dtype=float)
        out[: min(dim, diag.size)] = diag[: min(dim, diag.size)]
        if diag.size < dim and diag.size > 0:
            out[diag.size :] = diag[-1]
        return out
