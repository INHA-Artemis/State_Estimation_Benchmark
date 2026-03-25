from __future__ import annotations

from typing import Iterable

import numpy as np

from models import measurement_model
from models.state_model import state_dim, state_vector, zero_state


class UnscentedKalmanFilter:
    def __init__(
        self,
        pose_type: str = "2d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        sigma_point_config: dict | None = None,
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
        sigma_cfg = sigma_point_config or {}

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

        self.alpha = float(sigma_cfg.get("alpha", 0.3))
        self.beta = float(sigma_cfg.get("beta", 2.0))
        self.kappa = float(sigma_cfg.get("kappa", 0.0))
        self.lambda_ = self.alpha**2 * (self.dim + self.kappa) - self.dim

        self.x = zero_state(self.pose_type)
        self.P = np.eye(self.dim, dtype=float)
        self.initialized = False

    @classmethod
    def from_configs(cls, dataset_config: dict, ukf_config: dict) -> "UnscentedKalmanFilter":
        ukf = cls(
            pose_type=dataset_config.get("pose_type", "2d"),
            mode=dataset_config.get("mode", "fused"),
            motion_config=ukf_config.get("motion_model", {}),
            measurement_config=ukf_config.get("measurement_model", {}),
            sigma_point_config=ukf_config.get("sigma_points", {}),
        )
        init_cfg = ukf_config.get("initialization", {})
        ukf.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"))
        return ukf

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

        sigma_points, wm, wc = self._sigma_points(self.x, self.P)
        if control is None:
            propagated = sigma_points.copy()
        else:
            u = np.asarray(control, dtype=float).reshape(-1)
            propagated = np.vstack([self._transition_function(sigma, u, dt) for sigma in sigma_points])

        self.x = self._weighted_state_mean(propagated, wm)
        self.P = self._weighted_state_covariance(propagated, self.x, wc)
        self.P = self._nearest_spd(self.P + np.diag(np.clip(self.process_noise_diag, 1e-12, None)))
        return self.x.copy()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.x.copy()

        z = np.asarray(measurement, dtype=float).reshape(-1)
        if z.size != self.measurement_indices.size:
            raise ValueError("measurement size must match measurement indices.")

        sigma_points, wm, wc = self._sigma_points(self.x, self.P)
        sigma_measurements = np.vstack([measurement_model.h(sigma, indices=self.measurement_indices) for sigma in sigma_points])

        z_mean = np.sum(sigma_measurements * wm[:, None], axis=0)
        S = np.zeros((z.size, z.size), dtype=float)
        Pxz = np.zeros((self.dim, z.size), dtype=float)
        for idx in range(sigma_points.shape[0]):
            z_res = sigma_measurements[idx] - z_mean
            x_res = self._state_residual(sigma_points[idx], self.x)
            S += wc[idx] * np.outer(z_res, z_res)
            Pxz += wc[idx] * np.outer(x_res, z_res)

        R = np.diag(np.clip(self.measurement_noise_diag, 1e-12, None))
        S = self._nearest_spd(S + R)
        K = Pxz @ np.linalg.inv(S)

        self.x = self._normalize_angles(self.x + K @ (z - z_mean))
        self.P = self._nearest_spd(self.P - K @ S @ K.T)
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

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cov = self._nearest_spd(cov, eps=1e-12)
        scale = self.dim + self.lambda_
        scaled_cov = self._nearest_spd(scale * cov, eps=1e-12)
        try:
            chol = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(self._nearest_spd(scaled_cov + 1e-6 * np.eye(self.dim), eps=1e-9))

        sigma = np.zeros((2 * self.dim + 1, self.dim), dtype=float)
        sigma[0] = self._normalize_angles(mean)
        for i in range(self.dim):
            sigma[i + 1] = self._normalize_angles(mean + chol[:, i])
            sigma[self.dim + i + 1] = self._normalize_angles(mean - chol[:, i])

        wm = np.full(2 * self.dim + 1, 1.0 / (2.0 * scale), dtype=float)
        wc = np.full(2 * self.dim + 1, 1.0 / (2.0 * scale), dtype=float)
        wm[0] = self.lambda_ / scale
        wc[0] = self.lambda_ / scale + (1.0 - self.alpha**2 + self.beta)
        return sigma, wm, wc

    def _weighted_state_mean(self, sigma_points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.sum(sigma_points * weights[:, None], axis=0)
        for idx in self._angle_indices():
            s_term = np.sum(np.sin(sigma_points[:, idx]) * weights)
            c_term = np.sum(np.cos(sigma_points[:, idx]) * weights)
            mean[idx] = np.arctan2(s_term, c_term)
        return self._normalize_angles(mean)

    def _weighted_state_covariance(self, sigma_points: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> np.ndarray:
        cov = np.zeros((self.dim, self.dim), dtype=float)
        for idx in range(sigma_points.shape[0]):
            residual = self._state_residual(sigma_points[idx], mean)
            cov += weights[idx] * np.outer(residual, residual)
        return self._nearest_spd(cov)

    def _state_residual(self, x: np.ndarray, mean: np.ndarray) -> np.ndarray:
        residual = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
        for idx in self._angle_indices():
            residual[idx] = self._wrap_angle(residual[idx])
        return residual

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
