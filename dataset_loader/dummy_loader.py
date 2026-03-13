# [협업 주석]
# Goal: 실제 dataset parser 없이 PF pipeline을 빠르게 검증할 수 있는 synthetic loader를 제공한다.
# What it does: dummy trajectory를 생성하고 IMU/GPS noise를 주입해 DatasetSequence를 반환한다.
"""Synthetic placeholder loader for PF pipeline testing."""

from __future__ import annotations

from typing import Any

import numpy as np

from dataset_loader.dataset_base import DatasetLoader, DatasetSequence, DatasetStep
from utils.math_utils import wrap_angle


class DummySequenceLoader(DatasetLoader):
    """Generate synthetic trajectory with optional IMU/GPS observations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Goal:
            synthetic dataset 생성에 필요한 설정값을 loader에 저장한다.
        Input:
            cfg는 pose_type, sequence_length, dt, noise, sensor 사용 여부를 담은 dict이다.
        Output:
            없음. DummySequenceLoader instance field를 초기화한다.
        """
        self.cfg = cfg
        self.pose_type = str(cfg.get("pose_type", "2d")).lower()
        self.length = int(cfg.get("sequence_length", 300))
        self.dt = float(cfg.get("dt", 0.1))
        self.seed = int(cfg.get("seed", 0))

    def load_sequence(self) -> DatasetSequence:
        """
        Goal:
            synthetic trajectory와 optional IMU/GPS observation을 생성한다.
        Input:
            self는 seed와 sensor/noise 설정이 저장된 DummySequenceLoader instance이다.
        Output:
            생성된 DatasetStep 목록을 포함하는 DatasetSequence를 반환한다.
        """
        rng = np.random.default_rng(self.seed)
        is_6d = self.pose_type == "6d"
        state_dim = 6 if is_6d else 3

        imu_std = np.asarray(self.cfg.get("imu_noise_std", [0.05, 0.02]), dtype=float)
        gps_std = np.asarray(self.cfg.get("gps_noise_std", [0.7, 0.7]), dtype=float)
        gps_available = bool(self.cfg.get("use_gps", True))
        imu_available = bool(self.cfg.get("use_imu", True))

        x, y, yaw = 0.0, 0.0, 0.0
        steps: list[DatasetStep] = []

        for k in range(self.length):
            t = k * self.dt
            v_true = 1.0 + 0.2 * np.sin(0.05 * t)
            yaw_rate_true = 0.15 * np.cos(0.03 * t)

            x += v_true * self.dt * np.cos(yaw)
            y += v_true * self.dt * np.sin(yaw)
            yaw = float(wrap_angle(yaw + yaw_rate_true * self.dt))

            if is_6d:
                gt = np.array([x, y, 0.0, 0.0, 0.0, yaw], dtype=float)
            else:
                gt = np.array([x, y, yaw], dtype=float)

            imu = None
            if imu_available:
                imu = np.array(
                    [
                        v_true + rng.normal(0.0, imu_std[0]),
                        yaw_rate_true + rng.normal(0.0, imu_std[1]),
                    ],
                    dtype=float,
                )

            gps = None
            if gps_available:
                gps = gt[:2] + rng.normal(0.0, gps_std, size=2)

            steps.append(DatasetStep(t=t, dt=self.dt, imu=imu, gps=gps, gt_state=gt))

        return DatasetSequence(steps=steps, pose_type=self.pose_type, state_dim=state_dim)
