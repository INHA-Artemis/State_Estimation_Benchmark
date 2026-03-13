# [협업 주석]
# Goal: benchmark용 Particle Filter(PF) 핵심 로직을 modular하게 구현한다.
# What it does: particle 초기화, motion/measurement model 기반 predict/update, log-safe weight 처리,
# ESS 계산, systematic resampling, state/covariance 추정을 수행한다.
"""Particle Filter implementation."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from filters.base_filter import BaseFilter
from models.measurement_model import MeasurementModel
from models.motion_model import MotionModel
from utils.math_utils import weighted_mean_cov


class ParticleFilter(BaseFilter):
    """Generic particle filter with model-decoupled predict and update."""

    def __init__(
        self,
        num_particles: int,
        state_dim: int,
        motion_model: MotionModel,
        measurement_model: Optional[MeasurementModel] = None,
        resample_threshold_ratio: float = 0.5,
        angle_indices: Optional[Iterable[int]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Goal:
            ParticleFilter 실행에 필요한 particle storage와 dependency를 초기화한다.
        Input:
            num_particles, state_dim은 particle shape를 정의하고, motion_model/measurement_model은 predict/update model이다.
            resample_threshold_ratio, angle_indices, random_seed는 resampling과 random 동작을 제어한다.
        Output:
            없음. ParticleFilter instance field를 초기 상태로 설정한다.
        """
        self.num_particles = int(num_particles)
        self.state_dim = int(state_dim)
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.resample_threshold_ratio = float(resample_threshold_ratio)
        self.angle_indices = list(angle_indices or [])
        self.rng = np.random.default_rng(random_seed)

        self.particles = np.zeros((self.num_particles, self.state_dim), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)
        self._state = np.zeros(self.state_dim, dtype=float)
        self._cov = np.eye(self.state_dim, dtype=float)
        self.last_ess = float(self.num_particles)

    def reset(
        self,
        initial_mean: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
        particles: Optional[np.ndarray] = None,
    ) -> None:
        """
        Goal:
            particle 집합과 weight, state moment를 초기화한다.
        Input:
            initial_mean과 initial_cov는 Gaussian initialization에 사용되고, particles가 주어지면 explicit sample로 사용된다.
        Output:
            없음. 내부 particles와 weights가 재설정된다.
        """
        if particles is not None:
            arr = np.asarray(particles, dtype=float)
            if arr.shape != (self.num_particles, self.state_dim):
                raise ValueError(
                    f"particles must have shape {(self.num_particles, self.state_dim)}, got {arr.shape}"
                )
            self.particles = arr
        else:
            mean = np.zeros(self.state_dim, dtype=float) if initial_mean is None else np.asarray(initial_mean, dtype=float)
            cov = np.eye(self.state_dim, dtype=float) if initial_cov is None else np.asarray(initial_cov, dtype=float)
            if mean.shape != (self.state_dim,):
                raise ValueError(f"initial_mean must have shape {(self.state_dim,)}, got {mean.shape}")
            if cov.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"initial_cov must have shape {(self.state_dim, self.state_dim)}, got {cov.shape}"
                )
            self.particles = self.rng.multivariate_normal(mean, cov, size=self.num_particles)

        self.weights.fill(1.0 / self.num_particles)
        self._refresh_moments()

    def predict(self, u: Optional[np.ndarray], dt: float) -> None:
        """
        Goal:
            모든 particle을 MotionModel로 한 step propagate한다.
        Input:
            u는 optional control input이고, dt는 propagation에 사용할 time delta이다.
        Output:
            없음. 내부 particles와 추정 moment가 갱신된다.
        """
        self.particles = self.motion_model.propagate(self.particles, u, dt, noise=True)
        self._refresh_moments()

    def update(self, z: np.ndarray) -> None:
        """
        Goal:
            measurement likelihood를 이용해 particle weight를 업데이트하고 필요하면 resampling한다.
        Input:
            z는 현재 time step의 measurement vector이다.
        Output:
            없음. 내부 weights, particles, state moment, ESS 정보가 갱신된다.
        """
        if self.measurement_model is None:
            return

        z = np.asarray(z, dtype=float)
        if hasattr(self.measurement_model, "log_likelihood"):
            log_lik = self.measurement_model.log_likelihood(z, self.particles)
        else:
            lik = np.clip(self.measurement_model.likelihood(z, self.particles), 1e-300, None)
            log_lik = np.log(lik)

        log_weights = np.log(np.clip(self.weights, 1e-300, None)) + log_lik
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights = weights / total

        self.last_ess = self.effective_sample_size()
        if self.last_ess < self.resample_threshold_ratio * self.num_particles:
            self._systematic_resample()
            self.weights.fill(1.0 / self.num_particles)

        self._refresh_moments()

    def get_state(self) -> np.ndarray:
        """
        Goal:
            현재 weighted state estimate를 안전하게 반환한다.
        Input:
            self는 ParticleFilter instance이다.
        Output:
            내부 state estimate copy를 numpy array로 반환한다.
        """
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        """
        Goal:
            현재 weighted covariance estimate를 반환한다.
        Input:
            self는 ParticleFilter instance이다.
        Output:
            내부 covariance copy를 numpy array로 반환한다.
        """
        return self._cov.copy()

    def effective_sample_size(self) -> float:
        """
        Goal:
            현재 particle weight 분포의 Effective Sample Size를 계산한다.
        Input:
            self는 normalized weights를 가진 ParticleFilter instance이다.
        Output:
            ESS 값을 float로 반환한다.
        """
        return float(1.0 / np.sum(np.square(self.weights)))

    def _systematic_resample(self) -> None:
        """
        Goal:
            weight 분포에 따라 particle을 systematic resampling한다.
        Input:
            self는 particles와 weights를 가진 ParticleFilter instance이다.
        Output:
            없음. 내부 particles 배열이 resampled 결과로 교체된다.
        """
        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles
        cumulative = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indices]

    def _refresh_moments(self) -> None:
        """
        Goal:
            particle 집합으로부터 weighted mean과 covariance를 다시 계산한다.
        Input:
            self는 particles, weights, angle_indices를 가진 ParticleFilter instance이다.
        Output:
            없음. 내부 _state와 _cov가 최신 값으로 갱신된다.
        """
        self._state, self._cov = weighted_mean_cov(
            self.particles,
            self.weights,
            angle_indices=self.angle_indices,
        )
