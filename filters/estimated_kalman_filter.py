from __future__ import annotations

from typing import Iterable

import numpy as np

from models import measurement_model
from models.state_model import state_dim, zero_state
from utils.math_utils import fit_diag, nearest_spd
from utils.pose_filter_common import PoseFilterMixin


class ExtendedKalmanFilter(PoseFilterMixin):
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
        self.process_noise_diag = fit_diag(motion_cfg.get("process_noise_diag", np.zeros(self.dim)), self.dim)
        self.measurement_indices = np.asarray(
            meas_cfg.get("position_indices", list(range(default_meas_dim))),
            dtype=int,
        )
        self.measurement_noise_diag = fit_diag(
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
        self.P = nearest_spd(F @ self.P @ F.T + Q)
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
        S = nearest_spd(H @ self.P @ H.T + R)
        K = self.P @ H.T @ np.linalg.inv(S)

        I = np.eye(self.dim, dtype=float)
        self.x = self._normalize_angles(self.x + K @ innovation)
        self.P = nearest_spd((I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T)
        return self.x.copy()

    def _transition_jacobian(self, x: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(self.dim, dtype=float)
        if self.pose_type == "2d":
            speed = float(control[0])
            yaw = float(x[2])
            F[0, 2] = -speed * np.sin(yaw) * dt
            F[1, 2] = speed * np.cos(yaw) * dt
        return F
