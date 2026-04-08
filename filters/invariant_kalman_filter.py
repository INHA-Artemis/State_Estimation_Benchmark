from __future__ import annotations

from typing import Iterable

import numpy as np

from models import measurement_model, motion_model
from models.state_model import state_vector, zero_state
from utils.math_utils import exp_so3, fit_diag, fit_vector, nearest_spd, skew, wrap_angle
from utils.rotation_utils import rot_to_rpy, rpy_to_rot


class InvariantKalmanFilter:
    def __init__(
        self,
        pose_type: str = "2d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        initialization_config: dict | None = None,
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type not in ("2d", "3d"):
            raise ValueError("pose_type must be '2d' or '3d'.")

        self.pose_type = pose_type
        self.mode = mode

        motion_cfg = motion_config or {}
        measurement_cfg = measurement_config or {}
        init_cfg = initialization_config or {}

        if self.pose_type == "2d":
            self.error_dim = 3
            self.measurement_indices = np.asarray(
                measurement_cfg.get("position_indices", [0, 1]),
                dtype=int,
            )
        else:
            self.error_dim = 9
            self.measurement_indices = np.asarray(
                measurement_cfg.get("position_indices", [0, 1, 2]),
                dtype=int,
            )

        self.measurement_noise_diag = fit_diag(
            measurement_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
        )
        self.process_noise_diag = fit_diag(
            motion_cfg.get("process_noise_diag", np.full(self.error_dim, 1e-3)),
            self.error_dim,
        )
        self.linear_input_type = str(motion_cfg.get("linear_input_type", "velocity")).lower()
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(-1)
        if self.pose_type == "3d" and self.gravity.size != 3:
            raise ValueError("motion_model.gravity must contain 3 values for 3D pose.")

        self.P = np.eye(self.error_dim, dtype=float)
        self.initialized = False

        if self.pose_type == "2d":
            self.x = zero_state("2d")
        else:
            self.R = np.eye(3, dtype=float)
            self.v = np.zeros(3, dtype=float)
            self.p = np.zeros(3, dtype=float)

        self.initialize(
            mean=init_cfg.get("mean"),
            cov_diag=init_cfg.get("cov_diag"),
            velocity_mean=init_cfg.get("velocity_mean"),
        )

    @classmethod
    def from_configs(cls, dataset_config: dict, inekf_config: dict) -> "InvariantKalmanFilter":
        return cls(
            pose_type=dataset_config.get("pose_type", "2d"),
            mode=dataset_config.get("mode", "fused"),
            motion_config=inekf_config.get("motion_model", {}),
            measurement_config=inekf_config.get("measurement_model", {}),
            initialization_config=inekf_config.get("initialization", {}),
        )

    def initialize(
        self,
        mean: Iterable[float] | None = None,
        cov_diag: Iterable[float] | None = None,
        velocity_mean: Iterable[float] | None = None,
    ) -> None:
        if self.pose_type == "2d":
            if mean is None:
                mean_vec = zero_state("2d")
            else:
                mean_vec = state_vector(fit_vector(np.asarray(mean, dtype=float).reshape(-1), 3), "2d")
            self.x = self._normalize_angles(mean_vec)
        else:
            pose_mean = self._fit_pose_mean(mean)
            self.p = pose_mean[0:3]
            self.R = rpy_to_rot(pose_mean[3:6])
            if velocity_mean is None:
                self.v = np.zeros(3, dtype=float)
            else:
                self.v = fit_vector(np.asarray(velocity_mean, dtype=float).reshape(-1), 3)

        cov = fit_diag(np.ones(self.error_dim) * 1e-3 if cov_diag is None else cov_diag, self.error_dim)
        self.P = np.diag(np.clip(cov, 1e-12, None)).astype(float)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()

        if control is None:
            Phi = np.eye(self.error_dim, dtype=float)
        elif self.pose_type == "2d":
            Phi = self._predict_2d(np.asarray(control, dtype=float).reshape(-1), dt)
        else:
            Phi = self._predict_3d(np.asarray(control, dtype=float).reshape(-1), dt)

        Q = np.diag(np.clip(self.process_noise_diag, 1e-12, None))
        self.P = nearest_spd(Phi @ self.P @ Phi.T + Q)
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()

        z = np.asarray(measurement, dtype=float).reshape(-1)
        if z.size != self.measurement_indices.size:
            raise ValueError("measurement size must match measurement indices.")

        if self.pose_type == "2d":
            z_pred = measurement_model.h(self.x, indices=self.measurement_indices)
            H = np.zeros((z.size, self.error_dim), dtype=float)
            for row, idx in enumerate(self.measurement_indices):
                H[row, idx] = 1.0

            innovation = z - z_pred
            Rm = np.diag(np.clip(self.measurement_noise_diag, 1e-12, None))
            S = nearest_spd(H @ self.P @ H.T + Rm)
            K = self.P @ H.T @ np.linalg.inv(S)
            delta = K @ innovation

            self.x[0:2] += delta[0:2]
            self.x[2] = wrap_angle(self.x[2] + delta[2])
            I = np.eye(self.error_dim, dtype=float)
            self.P = nearest_spd((I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T)
            return self.estimate_pose()

        pose = self.estimate_pose()
        z_pred = measurement_model.h(pose, indices=self.measurement_indices)
        H = np.zeros((z.size, self.error_dim), dtype=float)
        for row, idx in enumerate(self.measurement_indices):
            if idx not in (0, 1, 2):
                raise ValueError("3D InEKF measurement indices must refer to position components [0, 1, 2].")
            H[row, 6 + idx] = 1.0

        innovation = z - z_pred
        Rm = np.diag(np.clip(self.measurement_noise_diag, 1e-12, None))
        S = nearest_spd(H @ self.P @ H.T + Rm)
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ innovation
        self._apply_error_3d(delta)

        I = np.eye(self.error_dim, dtype=float)
        self.P = nearest_spd((I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T)
        return self.estimate_pose()

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
        if self.pose_type == "2d":
            return self._normalize_angles(self.x.copy())
        return np.concatenate([self.p.copy(), rot_to_rpy(self.R)])

    def _predict_2d(self, control: np.ndarray, dt: float) -> np.ndarray:
        if control.size < 2:
            raise ValueError("2D control must contain at least [speed, yaw_rate].")

        speed = float(control[0])
        yaw_rate = float(control[1])
        yaw = float(self.x[2])
        dtheta = yaw_rate * dt

        yaw_mid = yaw + 0.5 * dtheta
        self.x[0] += speed * np.cos(yaw_mid) * dt
        self.x[1] += speed * np.sin(yaw_mid) * dt
        self.x[2] = wrap_angle(yaw + dtheta)

        Phi = np.eye(3, dtype=float)
        Phi[0:2, 0:2] = self._rot2(-dtheta)
        Phi[0, 2] = -speed * np.sin(yaw_mid) * dt
        Phi[1, 2] = speed * np.cos(yaw_mid) * dt
        return Phi

    def _predict_3d(self, control: np.ndarray, dt: float) -> np.ndarray:
        if control.size < 6:
            raise ValueError("3D control must contain at least 6 values.")

        linear = control[0:3]
        angular = control[3:6]
        dR = exp_so3(angular * dt)

        Phi = np.eye(9, dtype=float)
        Phi[0:3, 0:3] = dR.T
        Phi[6:9, 3:6] = np.eye(3, dtype=float) * dt

        if self.linear_input_type == "acceleration":
            prev_pose_state = np.concatenate([self.p, self.v, rot_to_rpy(self.R)])
            predicted = motion_model.f(prev_pose_state, control, dt=dt, g=self.gravity)
            accel_world = self.R @ linear + self.gravity
            self.p = predicted[0:3]
            self.v = predicted[3:6]
            self.R = rpy_to_rot(predicted[6:9])
            Phi[3:6, 0:3] = -skew(accel_world) * dt
            Phi[6:9, 0:3] = -0.5 * skew(accel_world) * (dt**2)
            return Phi

        self.R = self.R @ dR
        self.v = linear.copy()
        self.p = self.p + self.v * dt
        return Phi

    def _apply_error_3d(self, delta: np.ndarray) -> None:
        self.R = self.R @ exp_so3(delta[0:3])
        self.v = self.v + delta[3:6]
        self.p = self.p + delta[6:9]

    @staticmethod
    def _rot2(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _fit_pose_mean(self, mean: Iterable[float] | None) -> np.ndarray:
        if mean is None:
            return np.zeros(6, dtype=float)

        vector = np.asarray(mean, dtype=float).reshape(-1)
        if vector.size >= 9:
            return np.concatenate([vector[0:3], vector[6:9]])
        if vector.size == 6:
            return vector.copy()

        out = np.zeros(6, dtype=float)
        out[: min(6, vector.size)] = vector[: min(6, vector.size)]
        return out

    def _normalize_angles(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        if self.pose_type == "2d":
            x[2] = wrap_angle(x[2])
        else:
            x[3:6] = np.array([wrap_angle(angle) for angle in x[3:6]], dtype=float)
        return x
