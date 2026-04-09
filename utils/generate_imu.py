from __future__ import annotations

import numpy as np


def _expand_motion_noise(values, pose_type: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("IMU noise/bias config must not be empty.")

    if pose_type == "2d":
        if arr.size == 1:
            return np.repeat(arr, 2)
        if arr.size == 2:
            return arr
        raise ValueError("For 2D synthetic IMU, use one shared value or [speed, yaw_rate].")

    if arr.size == 1:
        return np.repeat(arr, 6)
    if arr.size == 2:
        linear_std, angular_std = arr
        return np.array([linear_std, linear_std, linear_std, angular_std, angular_std, angular_std], dtype=float)
    if arr.size == 6:
        return arr
    raise ValueError("For 3D synthetic IMU, use one shared value, [linear, angular], or 6 axis-specific values.")


def _sample_noisy_controls(ideal_controls: np.ndarray, dataset_cfg: dict, pose_type: str) -> np.ndarray:
    seed = int(dataset_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    noise_std = _expand_motion_noise(dataset_cfg.get("imu_noise_std", 0.0), pose_type)
    bias_std = _expand_motion_noise(dataset_cfg.get("imu_bias_std", 0.0), pose_type)

    fixed_bias = rng.normal(loc=0.0, scale=bias_std, size=ideal_controls.shape[1])
    white_noise = rng.normal(loc=0.0, scale=noise_std, size=ideal_controls.shape)
    return ideal_controls + fixed_bias + white_noise


def generate_imu_controls(dataset_cfg: dict, pose_type: str) -> tuple[np.ndarray, np.ndarray]:
    n = int(dataset_cfg.get("sequence_length", 300))
    dt = float(dataset_cfg.get("dt", 0.1))

    if pose_type == "2d":
        gt = np.zeros((n, 3), dtype=float)
        ideal_controls = np.zeros((n, 2), dtype=float)
        ideal_controls[:, 0] = 1.0 + 0.2 * np.sin(np.linspace(0.0, 6.0, n))
        ideal_controls[:, 1] = 0.12 * np.sin(np.linspace(0.0, 4.0, n))

        for k in range(1, n):
            x, y, yaw = gt[k - 1]
            speed, yaw_rate = ideal_controls[k]
            yaw = yaw + yaw_rate * dt
            x = x + speed * np.cos(yaw) * dt
            y = y + speed * np.sin(yaw) * dt
            gt[k] = np.array([x, y, yaw], dtype=float)

        noisy_controls = _sample_noisy_controls(ideal_controls, dataset_cfg, pose_type)
        return noisy_controls, gt

    gt = np.zeros((n, 6), dtype=float)
    ideal_controls = np.zeros((n, 6), dtype=float)
    ideal_controls[:, 0] = 0.8 + 0.2 * np.sin(np.linspace(0.0, 5.0, n))
    ideal_controls[:, 1] = 0.3 * np.cos(np.linspace(0.0, 3.0, n))
    ideal_controls[:, 2] = 0.1 * np.sin(np.linspace(0.0, 2.0, n))
    ideal_controls[:, 5] = 0.08 * np.sin(np.linspace(0.0, 4.0, n))

    for k in range(1, n):
        gt[k] = gt[k - 1] + ideal_controls[k] * dt

    noisy_controls = _sample_noisy_controls(ideal_controls, dataset_cfg, pose_type)
    return noisy_controls, gt
