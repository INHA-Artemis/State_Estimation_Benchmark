from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from models import measurement_model
from filters.particle_filter_resampling_algo import (
    multinomial_resample,
    residual_resample,
    stratified_resample,
    systematic_resample,
)
from models.state_model import state_dim, state_vector, zero_state
from utils.math_utils import fit_diag, fit_vector


class ParticleFilter:
    """간단한 NumPy 기반 Particle Filter."""

    def __init__(
        self,
        pose_type: str = "2d",
        mode: str = "fused",  # imu_only, gnss_only, fused
        num_particles: int = 500,
        resample_threshold_ratio: float = 0.5,
        seed: Optional[int] = None,
        motion_config: Optional[dict] = None,
        measurement_config: Optional[dict] = None,
        resampling_method: str = "systematic",
    ) -> None:
        # pose 타입 정규화 및 검증
        if pose_type == "6d":  # legacy alias
            pose_type = "3d"
        if pose_type not in ("2d", "3d"):
            raise ValueError("pose_type must be '2d' or '3d'.")

        # 기본 설정값
        self.pose_type = pose_type
        self.mode = mode
        self.dim = state_dim(pose_type)
        self.num_particles = int(num_particles)
        self.threshold = max(1, int(self.num_particles * float(resample_threshold_ratio)))
        self.rng = np.random.default_rng(seed)
        self.resampling_method = str(resampling_method).strip().lower()
        self._resamplers = {
            "multinomial": multinomial_resample,
            "residual": residual_resample,
            "stratified": stratified_resample,
            "systematic": systematic_resample,
        }
        if self.resampling_method not in self._resamplers:
            available = sorted(self._resamplers)
            raise ValueError(f"Unknown resampling_method: {self.resampling_method}. Available: {available}")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}

        # 프로세스 노이즈(상태 차원에 맞춰 보정)
        self.process_noise_diag = fit_diag(motion_cfg.get("process_noise_diag", np.zeros(self.dim)), self.dim, fill_missing="zero")
        self.linear_input_type = str(motion_cfg.get("linear_input_type", "velocity")).lower()
        if self.linear_input_type not in {"velocity", "acceleration"}:
            raise ValueError("motion_model.linear_input_type must be 'velocity' or 'acceleration'.")
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(-1)
        if self.pose_type == "3d" and self.gravity.size != 3:
            raise ValueError("motion_model.gravity must contain 3 values for 3D pose.")

        # 측정 인덱스/노이즈 설정
        default_meas_dim = 2 if self.pose_type == "2d" else 3
        self.measurement_indices = np.asarray(
            meas_cfg.get("position_indices", list(range(default_meas_dim))), dtype=int
        )
        self.measurement_noise_diag = fit_diag(
            meas_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
            fill_missing="zero",
        )
        self.likelihood_model = str(meas_cfg.get("likelihood_model", "gaussian")).strip().lower()
        if self.likelihood_model not in {"gaussian", "gaussian_mixture"}:
            raise ValueError("measurement_model.likelihood_model must be 'gaussian' or 'gaussian_mixture'.")
        self.outlier_weight = float(np.clip(float(meas_cfg.get("outlier_weight", 0.05)), 0.0, 1.0))
        default_outlier_diag = np.maximum(self.measurement_noise_diag * 400.0, 1.0)
        self.outlier_noise_diag = fit_diag(
            meas_cfg.get("outlier_noise_diag", default_outlier_diag),
            self.measurement_indices.size,
            fill_missing="zero",
        )

        self.particles = np.repeat(zero_state(self.pose_type)[None, :], self.num_particles, axis=0)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)
        self.initialized = False

    @classmethod
    def from_configs(cls, dataset_config: dict, pf_config: dict) -> "ParticleFilter":
        # yaml config를 읽어 PF 객체 생성
        pf = cls(
            pose_type=dataset_config.get("pose_type", "2d"),
            mode=dataset_config.get("mode", "fused"),
            num_particles=pf_config.get("num_particles", 500),
            resample_threshold_ratio=pf_config.get("resample_threshold_ratio", 0.5),
            seed=pf_config.get("seed"),
            motion_config=pf_config.get("motion_model", {}),
            measurement_config=pf_config.get("measurement_model", {}),
            resampling_method=pf_config.get("resampling_method", "systematic"),
        )

        init_cfg = pf_config.get("initialization", {})
        pf.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"))
        return pf

    def initialize(
        self,
        mean: Optional[Iterable[float]] = None,
        cov_diag: Optional[Iterable[float]] = None,
    ) -> None:
        # 초기 평균 상태 설정
        if mean is None:
            mean_vec = zero_state(self.pose_type)
        else:
            mean_vec = state_vector(fit_vector(np.asarray(mean, dtype=float).reshape(-1), self.dim), self.pose_type)

        # 초기 공분산(대각) 설정
        if cov_diag is None:
            cov = np.zeros(self.dim, dtype=float)
        else:
            cov = fit_diag(cov_diag, self.dim, fill_missing="zero")

        # 입자 샘플링
        std = np.sqrt(np.clip(cov, 0.0, None))
        self.particles = mean_vec[None, :] + self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
        self._normalize_particle_angles()
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True

    def predict(self, control: Optional[Iterable[float]], dt: float) -> np.ndarray:
        # 초기화가 안 되어 있으면 자동 초기화
        if not self.initialized:
            self.initialize()

        # 입력 기반 상태 예측
        if control is not None:
            u = np.asarray(control, dtype=float).reshape(-1)

            # 2D control shortcut: [speed, yaw_rate]
            if self.dim == 3 and u.size == 2:
                speed, yaw_rate = u
                yaw = self.particles[:, 2]
                self.particles[:, 0] += speed * np.cos(yaw) * dt
                self.particles[:, 1] += speed * np.sin(yaw) * dt
                self.particles[:, 2] += yaw_rate * dt
            elif self.dim >= 9 and u.size >= 6:
                linear_cmd = u[0:3][None, :]
                angular_cmd = u[3:6][None, :]

                if self.dim >= 15:
                    linear_cmd = linear_cmd - self.particles[:, 9:12]
                    angular_cmd = angular_cmd - self.particles[:, 12:15]

                if self.linear_input_type == "acceleration":
                    self.particles[:, 3:6] += (linear_cmd + self.gravity[None, :]) * dt
                else:
                    self.particles[:, 3:6] = linear_cmd

                self.particles[:, 0:3] += self.particles[:, 3:6] * dt
                self.particles[:, 6:9] += angular_cmd * dt
            else:
                u = fit_vector(u, self.dim)
                self.particles += u[None, :] * dt

        # dt가 작은 샘플에서 노이즈가 과도하게 누적되지 않도록 시간 간격에 맞춰 스케일한다.
        std = np.sqrt(np.clip(self.process_noise_diag, 0.0, None)) * np.sqrt(max(float(dt), 1e-12))
        self.particles += self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
        self._normalize_particle_angles()
        return self.particles

    def measurement_update(self, measurement: Optional[Iterable[float]]) -> np.ndarray:
        # 측정이 없으면 업데이트 생략
        if measurement is None:
            return self.weights

        z = np.asarray(measurement, dtype=float).reshape(-1)
        if z.size != self.measurement_indices.size:
            raise ValueError("measurement size must match measurement indices.")

        # 혁신 계산: (예측 측정 - 실제 측정)
        z_hat = self.particles[:, self.measurement_indices]
        innovation = z_hat - z[None, :]

        log_likelihood = self._measurement_log_likelihood(innovation)
        log_likelihood -= np.max(log_likelihood)

        self.weights *= np.exp(log_likelihood)
        self.normalize()
        return self.weights

    def _measurement_log_likelihood(self, innovation: np.ndarray) -> np.ndarray:
        inlier_var = np.clip(self.measurement_noise_diag, 1e-12, None)
        inlier_log = _diagonal_gaussian_logpdf(innovation, inlier_var)

        if self.likelihood_model == "gaussian":
            return inlier_log

        outlier_var = np.clip(self.outlier_noise_diag, 1e-12, None)
        outlier_log = _diagonal_gaussian_logpdf(innovation, outlier_var)
        inlier_weight = np.clip(1.0 - self.outlier_weight, 1e-12, 1.0)
        outlier_weight = np.clip(self.outlier_weight, 1e-12, 1.0)
        return np.logaddexp(
            np.log(inlier_weight) + inlier_log,
            np.log(outlier_weight) + outlier_log,
        )

    def normalize(self) -> np.ndarray:
        # 가중치 정규화 (비정상 값이면 균등 가중치로 복구)
        w_sum = np.sum(self.weights)
        if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= w_sum
        return self.weights

    def effective_sample_size(self) -> float:
        # 유효 샘플 수(ESS)
        return 1.0 / np.sum(self.weights**2)

    def resample(self) -> None:
        indices = self._resamplers[self.resampling_method](self.weights, self.rng)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_pose(self) -> np.ndarray:
        # 가중 평균 상태를 pose 형태(2D: 3, 3D: 6)로 반환
        mean_state = np.average(self.particles, axis=0, weights=self.weights)

        if self.dim == 3:
            est = mean_state.copy()
            s = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
            c = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
            est[2] = np.arctan2(s, c)
            return est

        if self.dim >= 9:
            pose = np.concatenate([mean_state[0:3], mean_state[6:9]])
            angle_cols = [6, 7, 8]
            for pose_idx, col_idx in enumerate(angle_cols, start=3):
                s = np.average(np.sin(self.particles[:, col_idx]), weights=self.weights)
                c = np.average(np.cos(self.particles[:, col_idx]), weights=self.weights)
                pose[pose_idx] = np.arctan2(s, c)
            return pose

        est = mean_state.copy()
        for idx in self._angle_indices():
            s = np.average(np.sin(self.particles[:, idx]), weights=self.weights)
            c = np.average(np.cos(self.particles[:, idx]), weights=self.weights)
            est[idx] = np.arctan2(s, c)
        return est

    def step(
        self,
        control: Optional[Iterable[float]],
        measurement: Optional[Iterable[float]],
        dt: float,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        # 모드에 따라 predict/update 수행
        run_mode = self.mode if mode is None else mode

        if run_mode in ("imu_only", "fused"):
            self.predict(control, dt)
        if run_mode in ("gnss_only", "fused"):
            self.measurement_update(measurement)

        if self.effective_sample_size() < self.threshold:
            self.resample()

        return self.estimate_pose()

    def run(self, dataset, mode: Optional[str] = None) -> np.ndarray:
        """dataset: {control, measurement, dt} 키를 가진 dict iterable."""
        estimates = []
        for sample in dataset:
            control = sample.get("control")
            measurement = sample.get("measurement")
            dt = float(sample.get("dt", 1.0))
            estimates.append(self.step(control, measurement, dt, mode=mode))

        if not estimates:
            return np.zeros((0, 3 if self.pose_type == "2d" else 6), dtype=float)
        return np.vstack(estimates)

    def expected_measurement(self, state: Iterable[float]) -> np.ndarray:
        """단일 상태에서 기대 측정값 h(x) 계산."""
        return measurement_model.h(state, indices=self.measurement_indices)

    def _angle_indices(self) -> list[int]:
        if self.dim == 3:
            return [2]
        if self.dim >= 9:
            return [6, 7, 8]
        return [3, 4, 5]

    def _normalize_particle_angles(self) -> None:
        for idx in self._angle_indices():
            self.particles[:, idx] = np.arctan2(np.sin(self.particles[:, idx]), np.cos(self.particles[:, idx]))


def _diagonal_gaussian_logpdf(innovation: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.asarray(variance, dtype=float).reshape(1, -1)
    return -0.5 * (
        np.sum((innovation**2) / variance, axis=1)
        + np.sum(np.log(2.0 * np.pi * variance))
    )
