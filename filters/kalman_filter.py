from __future__ import annotations

from typing import Iterable

import numpy as np

from models import constant_velocity as cv
from utils.filter_math import diagonal_covariance, kalman_update
from utils.math_utils import fit_diag, fit_vector


class KalmanFilter:
    """Linear KF for the shared constant-velocity benchmark model."""

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
    ) -> None:
        self.pose_type = cv.normalize_pose_type(pose_type)
        self.mode = mode
        self.pos_dim = cv.position_dim(self.pose_type)
        self.dim = cv.state_dim(self.pose_type)

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        self.process_noise_diag = fit_diag(motion_cfg.get("process_noise_diag", np.full(self.dim, 1e-3)), self.dim)
        self.control_input_type = str(motion_cfg.get("control_input_type", "none")).lower()
        self.gravity = fit_vector(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), self.pos_dim)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", np.zeros(self.pos_dim)), self.pos_dim)
        self.measurement_indices = np.asarray(meas_cfg.get("position_indices", list(range(self.pos_dim))), dtype=int)
        self.measurement_noise_diag = fit_diag(
            meas_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
        )

        self.x = np.zeros(self.dim, dtype=float)
        self.P = np.eye(self.dim, dtype=float)
        self.F = np.eye(self.dim, dtype=float)
        self.Q = diagonal_covariance(self.process_noise_diag)
        self.H = cv.measurement_matrix(self.measurement_indices, self.dim)
        self.R = diagonal_covariance(self.measurement_noise_diag)
        self.y = np.zeros(self.measurement_indices.size, dtype=float)
        self.S = np.eye(self.measurement_indices.size, dtype=float)
        self.K = np.zeros((self.dim, self.measurement_indices.size), dtype=float)
        self.initialized = False

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "KalmanFilter":
        cfg = compare_config.get("kalman_filter", compare_config)
        kf = cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
        )
        kf.initialize(cfg.get("initialization", {}).get("mean"), cfg.get("initialization", {}).get("cov_diag"))
        return kf

    def initialize(self, mean: Iterable[float] | None = None, cov_diag: Iterable[float] | None = None) -> None:
        self.x = cv.fit_state(mean, self.pose_type)
        self.P = diagonal_covariance(fit_diag(np.ones(self.dim) if cov_diag is None else cov_diag, self.dim))
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        # KF-P1: build the linear transition and process covariance.
        self.F = cv.transition_matrix(self.pos_dim, dt)
        self.Q = diagonal_covariance(self.process_noise_diag)
        # KF-P2: predict state mean, x_k^- = F_k x_{k-1} + b_k.
        self.x = self.F @ self.x + cv.control_vector(
            control, self.pos_dim, dt, self.control_input_type, self.accel_bias, self.gravity
        )
        # KF-P3: predict covariance, P_k^- = F_k P_{k-1} F_k^T + Q_k.
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        # KF-U1: validate measurement z_k.
        z = cv.validate_measurement(measurement, self.measurement_indices.size)
        # KF-U2: build linear measurement matrix and measurement covariance.
        self.H = cv.measurement_matrix(self.measurement_indices, self.dim)
        self.R = diagonal_covariance(self.measurement_noise_diag)
        # KF-U3: correct with y_k, S_k, K_k, x_k, and Joseph-form P_k.
        self.x, self.P, self.y, self.S, self.K = kalman_update(self.x, self.P, z, self.H, self.R)
        return self.estimate_pose()

    def step(
        self,
        control: Iterable[float] | None,
        measurement: Iterable[float] | None,
        dt: float,
        mode: str | None = None,
    ) -> np.ndarray:
        run_mode = self.mode if mode is None else mode
        if run_mode in {"imu_only", "fused"}:
            # KF-S1: run prediction when control/IMU is enabled.
            self.predict(control, dt)
        if run_mode in {"gnss_only", "fused"}:
            # KF-S2: run measurement update when GNSS/position is enabled.
            self.measurement_update(measurement)
        # KF-S3: expose the benchmark pose format.
        return self.estimate_pose()

    def run(self, dataset, mode: str | None = None) -> np.ndarray:
        estimates = [
            self.step(sample.get("control"), sample.get("measurement"), float(sample.get("dt", 1.0)), mode=mode)
            for sample in dataset
        ]
        return np.vstack(estimates) if estimates else np.zeros((0, 3 if self.pose_type == "2d" else 6), dtype=float)

    def estimate_pose(self) -> np.ndarray:
        return cv.pose_from_state(self.x, self.pose_type)
