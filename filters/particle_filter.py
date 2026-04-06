from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from models import measurement_model
from models.state_model import state_dim, state_vector, zero_state


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
    ) -> None:
        # pose 타입 정규화 및 검증
        if pose_type == "6d":
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

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}

        # 프로세스 노이즈(상태 차원에 맞춰 보정)
        self.process_noise_diag = self._fit_diag(motion_cfg.get("process_noise_diag", np.zeros(self.dim)), self.dim)

        # 측정 인덱스/노이즈 설정
        default_meas_dim = 2 if self.pose_type == "2d" else 3
        self.measurement_indices = np.asarray(
            meas_cfg.get("position_indices", list(range(default_meas_dim))), dtype=int
        )
        self.measurement_noise_diag = self._fit_diag(
            meas_cfg.get("measurement_noise_diag", np.ones(self.measurement_indices.size)),
            self.measurement_indices.size,
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
            mean_vec = state_vector(self._fit_vector(np.asarray(mean, dtype=float).reshape(-1), self.dim), self.pose_type)

        # 초기 공분산(대각) 설정
        if cov_diag is None:
            cov = np.zeros(self.dim, dtype=float)
        else:
            cov = self._fit_diag(cov_diag, self.dim)

        # 입자 샘플링
        std = np.sqrt(np.clip(cov, 0.0, None))
        self.particles = mean_vec[None, :] + self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
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
            else:
                u = self._fit_vector(u, self.dim)
                self.particles += u[None, :] * dt

        # dt가 작은 샘플에서 노이즈가 과도하게 누적되지 않도록 시간 간격에 맞춰 스케일한다.
        std = np.sqrt(np.clip(self.process_noise_diag, 0.0, None)) * np.sqrt(max(float(dt), 1e-12))
        self.particles += self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
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

        # 대각 공분산 가정 가우시안 likelihood
        var = np.clip(self.measurement_noise_diag, 1e-12, None)
        log_likelihood = -0.5 * np.sum((innovation**2) / var[None, :], axis=1)
        log_likelihood -= np.max(log_likelihood)

        self.weights *= np.exp(log_likelihood)
        self.normalize()
        return self.weights

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
        # Systematic resampling
        cumulative = np.cumsum(self.weights)
        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles
        indices = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_pose(self) -> np.ndarray:
        # 가중 평균 상태 추정
        est = np.average(self.particles, axis=0, weights=self.weights)
        # 각도 항목은 원형 평균 사용
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
        if run_mode in ("gps_only", "gnss_only", "fused"):
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
            return np.zeros((0, self.dim), dtype=float)
        return np.vstack(estimates)

    def expected_measurement(self, state: Iterable[float]) -> np.ndarray:
        """단일 상태에서 기대 측정값 h(x) 계산."""
        return measurement_model.h(state, indices=self.measurement_indices)

    def _angle_indices(self) -> list[int]:
        if self.dim == 3:
            return [2]
        return [3, 4, 5]

    @staticmethod
    def _fit_vector(values: np.ndarray, dim: int) -> np.ndarray:
        # 벡터 길이를 상태 차원에 맞춰 자르거나 0으로 패딩
        if values.size == dim:
            return values
        out = np.zeros(dim, dtype=float)
        out[: min(dim, values.size)] = values[: min(dim, values.size)]
        return out

    @staticmethod
    def _fit_diag(values: Iterable[float], dim: int) -> np.ndarray:
        # 대각값 길이를 목표 차원에 맞춤
        diag = np.asarray(values, dtype=float).reshape(-1)
        if diag.size == dim:
            return diag
        if diag.size == 1:
            return np.full(dim, float(diag.item()), dtype=float)
        out = np.zeros(dim, dtype=float)
        out[: min(dim, diag.size)] = diag[: min(dim, diag.size)]
        return out
