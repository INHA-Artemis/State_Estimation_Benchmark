from __future__ import annotations

from typing import Iterable

import numpy as np

from models import invariant_inekf as lie
from utils.filter_math import diagonal_covariance, kalman_update
from utils.math_utils import fit_diag, fit_vector


class InvariantKalmanFilter15D:
    """SE_2(3) InEKF with 15D error state [dR, dv, dp, dbg, dba]."""

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        initialization_config: dict | None = None,
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type != "3d":
            raise ValueError("InvariantKalmanFilter15D supports only 3d pose.")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        init_cfg = initialization_config or {}

        self.pose_type = pose_type
        self.mode = mode
        self.error_dim = 15
        self.use_imu_velocity = bool(motion_cfg.get("use_imu_velocity", True))
        self.use_imu_rotation = bool(motion_cfg.get("use_imu_rotation", True))
        self.translation_input_frame = str(motion_cfg.get("translation_input_frame", "body"))
        self.translation_input_type = str(motion_cfg.get("translation_input_type", "acceleration"))
        self.rotation_input_type = str(motion_cfg.get("rotation_input_type", "rate"))
        self.rotation_representation = str(motion_cfg.get("rotation_representation", "rotvec"))
        self.velocity_blend = float(motion_cfg.get("velocity_blend", 0.0))
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.process_noise_diag = fit_diag(
            motion_cfg.get(
                "process_noise_diag",
                [1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5],
            ),
            self.error_dim,
        )
        self.measurement_noise_diag = fit_diag(
            meas_cfg.get("measurement_noise_diag", [1.0, 1.0, 1.0]),
            3,
        )
        self.innovation_gate_m = float(meas_cfg.get("innovation_gate_m", 0.0))
        self.mahalanobis_gate = float(meas_cfg.get("mahalanobis_gate", 0.0))

        self.Rot = np.eye(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.p = np.zeros(3, dtype=float)
        self.gyro_bias = fit_vector(motion_cfg.get("gyro_bias", [0.0, 0.0, 0.0]), 3)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), 3)
        self.P = np.eye(self.error_dim, dtype=float)
        self.Phi = np.eye(self.error_dim, dtype=float)
        self.Q = diagonal_covariance(self.process_noise_diag)
        self.H = np.zeros((3, self.error_dim), dtype=float)
        self.H[:, 6:9] = np.eye(3)
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        self.innovation = np.zeros(3, dtype=float)
        self.K = np.zeros((self.error_dim, 3), dtype=float)
        self.delta = np.zeros(self.error_dim, dtype=float)
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.initialized = False
        self.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"), init_cfg.get("velocity_mean"))

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "InvariantKalmanFilter15D":
        cfg = compare_config.get("invariant_kalman_filter_15d", compare_config.get("invariant_kalman_filter", compare_config))
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
        )

    def initialize(
        self,
        mean: Iterable[float] | None = None,
        cov_diag: Iterable[float] | None = None,
        velocity_mean: Iterable[float] | None = None,
    ) -> None:
        pose = fit_vector(np.zeros(6) if mean is None else np.asarray(mean, dtype=float).reshape(-1), 6)
        self.p, self.Rot = lie.pose_to_state(pose)
        self.v = fit_vector(np.zeros(3) if velocity_mean is None else np.asarray(velocity_mean, dtype=float).reshape(-1), 3)
        cov = fit_diag(np.ones(self.error_dim) * 1e-3 if cov_diag is None else cov_diag, self.error_dim)
        self.P = diagonal_covariance(cov)
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        if control is None:
            return self.estimate_pose()

        u = np.asarray(control, dtype=float).reshape(-1)
        if u.size < 6:
            raise ValueError("3D InEKF control must contain [ax, ay, az, gx, gy, gz].")
        dt = float(dt)
        dt = max(dt, 0.0)
        dt2 = dt * dt

        R_prev = self.Rot
        v_prev = self.v.copy()
        w = u[3:6] - self.gyro_bias
        a = u[0:3] - self.accel_bias
        phi = w * dt
        G1 = lie.so3_left_jacobian(phi)
        G2 = _gamma2(phi)

        if self.use_imu_rotation:
            if self.rotation_input_type == "rate":
                rot_vec = phi
            elif self.rotation_input_type == "increment":
                rot_vec = w
            else:
                raise ValueError(f"Unsupported rotation_input_type: {self.rotation_input_type}")

            if self.rotation_representation == "rotvec":
                self.Rot = R_prev @ lie.so3_exp(rot_vec)
            elif self.rotation_representation == "euler":
                self.Rot = lie.rpy_to_rot(lie.rot_to_rpy(R_prev) + rot_vec)
            else:
                raise ValueError(f"Unsupported rotation_representation: {self.rotation_representation}")

        if self.use_imu_velocity:
            if self.translation_input_frame == "body":
                trans_world = R_prev @ G1 @ a if self.translation_input_type == "acceleration" else R_prev @ a
            elif self.translation_input_frame == "world":
                trans_world = a
            else:
                raise ValueError(f"Unsupported translation_input_frame: {self.translation_input_frame}")

            if self.translation_input_type == "acceleration":
                accel_world = trans_world + self.gravity
                if self.translation_input_frame == "body":
                    self.p = self.p + v_prev * dt + R_prev @ G2 @ a * dt2 + 0.5 * self.gravity * dt2
                    self.v = self.v + accel_world * dt
                else:
                    self.p = self.p + v_prev * dt + 0.5 * accel_world * dt2
                    self.v = self.v + accel_world * dt
            elif self.translation_input_type == "velocity":
                self.v = self.velocity_blend * self.v + (1.0 - self.velocity_blend) * trans_world
                self.p = self.p + self.v * dt
            elif self.translation_input_type == "increment":
                self.p = self.p + trans_world
                if dt > 1e-12:
                    inferred_v = trans_world / dt
                    self.v = self.velocity_blend * self.v + (1.0 - self.velocity_blend) * inferred_v
            else:
                raise ValueError(f"Unsupported translation_input_type: {self.translation_input_type}")
        else:
            self.p = self.p + self.v * dt

        A = np.zeros((self.error_dim, self.error_dim), dtype=float)
        A[0:3, 0:3] = -lie.skew(w)
        A[0:3, 9:12] = -np.eye(3)
        A[3:6, 0:3] = -lie.skew(a)
        A[3:6, 3:6] = -lie.skew(w)
        A[3:6, 12:15] = -np.eye(3)
        A[6:9, 3:6] = np.eye(3)
        A[6:9, 6:9] = -lie.skew(w)

        self.Phi = _matrix_exp(A * dt)
        self.Q = diagonal_covariance(self.process_noise_diag)
        predicted = self.Phi @ self.P @ self.Phi.T + self.Phi @ self.Q @ self.Phi.T * max(dt, 1e-9)
        self.P = 0.5 * (predicted + predicted.T)
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self.innovation = z - self.p
        if self._reject_measurement():
            return self.estimate_pose()

        self.H = np.zeros((3, self.error_dim), dtype=float)
        self.H[:, 6:9] = np.eye(3)
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        _, P_update, self.innovation, self.S, self.K = kalman_update(
            np.zeros(self.error_dim, dtype=float), self.P, self.innovation, self.H, self.Rm
        )
        self.delta = self.K @ self.innovation
        dX = lie.se23_exp(self.delta[:9])
        self.Rot, self.v, self.p = lie.from_matrix(dX @ lie.as_matrix(self.Rot, self.v, self.p))
        self.gyro_bias = self.gyro_bias + self.delta[9:12]
        self.accel_bias = self.accel_bias + self.delta[12:15]
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.P = 0.5 * (P_update + P_update.T)
        return self.estimate_pose()

    def velocity_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self.innovation = z - self.v
        if self.innovation_gate_m > 0.0 and np.linalg.norm(self.innovation) > self.innovation_gate_m:
            return self.estimate_pose()

        H = np.zeros((3, self.error_dim), dtype=float)
        H[:, 3:6] = np.eye(3)
        Rm = diagonal_covariance(fit_diag(self.measurement_noise_diag, 3))
        S = H @ self.P @ H.T + Rm + 1e-12 * np.eye(3)
        if self.mahalanobis_gate > 0.0:
            maha = float(self.innovation.T @ np.linalg.solve(S, self.innovation))
            if maha > self.mahalanobis_gate:
                return self.estimate_pose()

        _, P_update, self.innovation, self.S, self.K = kalman_update(
            np.zeros(self.error_dim, dtype=float), self.P, self.innovation, H, Rm
        )
        self.delta = self.K @ self.innovation
        dX = lie.se23_exp(self.delta[:9])
        self.Rot, self.v, self.p = lie.from_matrix(dX @ lie.as_matrix(self.Rot, self.v, self.p))
        self.gyro_bias = self.gyro_bias + self.delta[9:12]
        self.accel_bias = self.accel_bias + self.delta[12:15]
        self.X = lie.as_matrix(self.Rot, self.v, self.p)
        self.P = 0.5 * (P_update + P_update.T)
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
            self.predict(control, dt)
        if run_mode in {"gnss_only", "fused"}:
            self.measurement_update(measurement)
        return self.estimate_pose()

    def estimate_pose(self) -> np.ndarray:
        return lie.pose_from_state(self.Rot, self.p)

    def _reject_measurement(self) -> bool:
        if self.innovation_gate_m > 0.0 and np.linalg.norm(self.innovation) > self.innovation_gate_m:
            return True
        self.H = np.zeros((3, self.error_dim), dtype=float)
        self.H[:, 6:9] = np.eye(3)
        self.Rm = diagonal_covariance(self.measurement_noise_diag)
        self.S = self.H @ self.P @ self.H.T + self.Rm + 1e-12 * np.eye(3)
        if self.mahalanobis_gate <= 0.0:
            return False
        maha = float(self.innovation.T @ np.linalg.solve(self.S, self.innovation))
        return maha > self.mahalanobis_gate


def _gamma2(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = lie.skew(phi)
    if theta < 1e-8:
        return 0.5 * np.eye(3) + (1.0 / 6.0) * K + (1.0 / 24.0) * (K @ K)
    return (
        0.5 * np.eye(3)
        + ((theta - np.sin(theta)) / theta**3) * K
        + ((theta**2 + 2.0 * np.cos(theta) - 2.0) / (2.0 * theta**4)) * (K @ K)
    )


def _matrix_exp(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix, ord=np.inf))
    scale = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0.5 else 0
    A = matrix / (2**scale)
    result = np.eye(A.shape[0], dtype=float)
    term = np.eye(A.shape[0], dtype=float)
    for order in range(1, 24):
        term = term @ A / float(order)
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-14:
            break
    for _ in range(scale):
        result = result @ result
    return result
