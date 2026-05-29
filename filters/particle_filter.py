from __future__ import annotations

from typing import Iterable

import numpy as np

from models import constant_velocity as cv
from utils.filter_math import diagonal_gaussian_logpdf
from utils.math_utils import fit_diag, fit_vector
from utils.resampling import resample_indices


class ParticleFilter:
    """Bootstrap PF over the shared constant-velocity benchmark model."""

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        num_particles: int = 1000,
        resample_threshold_ratio: float = 0.5,
        seed: int | None = None,
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        resampling_method: str = "systematic",
    ) -> None:
        self.pose_type = cv.normalize_pose_type(pose_type)
        self.mode = mode
        self.pos_dim = cv.position_dim(self.pose_type)
        self.dim = cv.state_dim(self.pose_type)
        self.num_particles = int(num_particles)
        self.threshold = max(1, int(self.num_particles * float(resample_threshold_ratio)))
        self.rng = np.random.default_rng(seed)

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        self.process_noise_diag = fit_diag(
            motion_cfg.get("process_noise_diag", np.full(self.dim, 1e-3)),
            self.dim,
            fill_missing="zero",
        )
        self.control_input_type = str(motion_cfg.get("control_input_type", "none")).lower()
        self.gravity = fit_vector(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), self.pos_dim)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", np.zeros(self.pos_dim)), self.pos_dim)
        self.measurement_indices = np.asarray(meas_cfg.get("position_indices", list(range(self.pos_dim))), dtype=int)
        self.measurement_noise_diag = fit_diag(
            meas_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
            fill_missing="zero",
        )
        self.measurement_rejuvenation = bool(meas_cfg.get("measurement_rejuvenation", False))
        self.measurement_rejuvenation_std = fit_diag(
            meas_cfg.get("measurement_rejuvenation_std", np.sqrt(np.clip(self.measurement_noise_diag, 1e-12, None))),
            self.measurement_indices.size,
            fill_missing="edge",
        )
        self.resampling_method = str(resampling_method).lower()
        self.particles = np.zeros((self.num_particles, self.dim), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)
        self.log_likelihood = np.zeros(self.num_particles, dtype=float)
        self.last_position_covariance = np.eye(self.pos_dim, dtype=float)
        self.initialized = False

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "ParticleFilter":
        cfg = compare_config.get("particle_filter", compare_config)
        pf = cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            num_particles=cfg.get("num_particles", 1000),
            resample_threshold_ratio=cfg.get("resample_threshold_ratio", 0.5),
            seed=cfg.get("seed"),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            resampling_method=cfg.get("resampling_method", "systematic"),
        )
        pf.initialize(cfg.get("initialization", {}).get("mean"), cfg.get("initialization", {}).get("cov_diag"))
        return pf

    def initialize(self, mean: Iterable[float] | None = None, cov_diag: Iterable[float] | None = None) -> None:
        mean_vec = cv.fit_state(mean, self.pose_type)
        cov = fit_diag(np.zeros(self.dim) if cov_diag is None else cov_diag, self.dim, fill_missing="zero")
        # PF-I1: sample particles from the configured initial Gaussian.
        self.particles = mean_vec[None, :] + self.rng.normal(0.0, np.sqrt(np.clip(cov, 0.0, None)), size=(self.num_particles, self.dim))
        # PF-I2: start with uniform particle weights.
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        # PF-P1: propagate each particle with the constant-velocity model.
        F = cv.transition_matrix(self.pos_dim, dt)
        b = cv.control_vector(control, self.pos_dim, dt, self.control_input_type, self.accel_bias, self.gravity)
        std = np.sqrt(np.clip(self.process_noise_diag, 0.0, None)) * np.sqrt(max(float(dt), 1e-12))
        self.particles = self.particles @ F.T + b[None, :]
        # PF-P2: add process noise to represent q(x_k | x_{k-1}, u_k).
        self.particles += self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
        return self.particles

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.weights
        # PF-U1: evaluate measurement residuals for all particles.
        z = cv.validate_measurement(measurement, self.measurement_indices.size)
        innovation = self.particles[:, self.measurement_indices] - z[None, :]
        # PF-U2: multiply weights by the Gaussian likelihood p(z_k | x_k^i).
        self.log_likelihood = diagonal_gaussian_logpdf(innovation, np.clip(self.measurement_noise_diag, 1e-12, None))
        self.weights *= np.exp(self.log_likelihood - np.max(self.log_likelihood))
        # PF-U3: normalize weights so sum_i w_k^i = 1.
        return self.normalize()

    def normalize(self) -> np.ndarray:
        total = float(np.sum(self.weights))
        self.weights[:] = 1.0 / self.num_particles if not np.isfinite(total) or total <= 0.0 else self.weights / total
        return self.weights

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def resample(self) -> None:
        # PF-R1: resample particles and reset weights after sample impoverishment check.
        self.particles = self.particles[resample_indices(self.resampling_method, self.weights, self.rng)]
        self.weights.fill(1.0 / self.num_particles)

    def step(
        self,
        control: Iterable[float] | None,
        measurement: Iterable[float] | None,
        dt: float,
        mode: str | None = None,
    ) -> np.ndarray:
        run_mode = self.mode if mode is None else mode
        if run_mode in {"imu_only", "fused"}:
            # PF-S1: run particle prediction when control/IMU is enabled.
            self.predict(control, dt)
        if run_mode in {"gnss_only", "fused"}:
            # PF-S2: run likelihood update when GNSS/position is enabled.
            self.measurement_update(measurement)
        if self.effective_sample_size() < self.threshold:
            # PF-S4: resample only when effective sample size is below threshold.
            self.resample()
        if measurement is not None and self.measurement_rejuvenation:
            self.rejuvenate_from_measurement(measurement)
        estimate = self.estimate_pose()
        self.last_position_covariance = self.position_covariance()
        return estimate

    def run(self, dataset, mode: str | None = None) -> np.ndarray:
        estimates = [
            self.step(sample.get("control"), sample.get("measurement"), float(sample.get("dt", 1.0)), mode=mode)
            for sample in dataset
        ]
        return np.vstack(estimates) if estimates else np.zeros((0, 3 if self.pose_type == "2d" else 6), dtype=float)

    def estimate_pose(self) -> np.ndarray:
        return cv.pose_from_state(np.average(self.particles, axis=0, weights=self.weights), self.pose_type)

    def position_covariance(self) -> np.ndarray:
        positions = self.particles[:, : self.pos_dim]
        mean = np.average(positions, axis=0, weights=self.weights)
        centered = positions - mean[None, :]
        return centered.T @ (centered * self.weights[:, None])

    def rejuvenate_from_measurement(self, measurement: Iterable[float]) -> None:
        z = cv.validate_measurement(measurement, self.measurement_indices.size)
        std = np.sqrt(np.clip(self.measurement_noise_diag, 1e-12, None))
        configured = np.asarray(self.measurement_rejuvenation_std, dtype=float).reshape(-1)
        std = np.maximum(std, configured)
        self.particles[:, self.measurement_indices] = z[None, :] + self.rng.normal(
            0.0,
            std,
            size=(self.num_particles, self.measurement_indices.size),
        )
        self.weights.fill(1.0 / self.num_particles)
