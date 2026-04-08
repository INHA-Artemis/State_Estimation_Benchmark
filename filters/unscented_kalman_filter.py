from __future__ import annotations

from typing import Iterable

import numpy as np

from models import measurement_model
from models.state_model import state_dim, zero_state
from utils.math_utils import fit_diag, nearest_spd, wrap_angle
from utils.pose_filter_common import PoseFilterMixin


class UnscentedKalmanFilter(PoseFilterMixin):
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
        self.process_noise_diag = fit_diag(motion_cfg.get("process_noise_diag", np.zeros(self.dim)), self.dim)
        self.measurement_indices = np.asarray(
            meas_cfg.get("position_indices", list(range(default_meas_dim))),
            dtype=int,
        )
        self.measurement_noise_diag = fit_diag(
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
        self.P = nearest_spd(self.P + np.diag(np.clip(self.process_noise_diag, 1e-12, None)))
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
        S = nearest_spd(S + R)
        K = Pxz @ np.linalg.inv(S)

        self.x = self._normalize_angles(self.x + K @ (z - z_mean))
        self.P = nearest_spd(self.P - K @ S @ K.T)
        return self.x.copy()

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cov = nearest_spd(cov, eps=1e-12)
        scale = self.dim + self.lambda_
        scaled_cov = nearest_spd(scale * cov, eps=1e-12)
        try:
            chol = np.linalg.cholesky(scaled_cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(nearest_spd(scaled_cov + 1e-6 * np.eye(self.dim), eps=1e-9))

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
        return nearest_spd(cov)

    def _state_residual(self, x: np.ndarray, mean: np.ndarray) -> np.ndarray:
        residual = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
        for idx in self._angle_indices():
            residual[idx] = wrap_angle(residual[idx])
        return residual
