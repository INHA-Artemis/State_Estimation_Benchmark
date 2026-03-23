from __future__ import annotations

import numpy as np


def generate_imu_controls(dataset_cfg: dict, pose_type: str) -> tuple[np.ndarray, np.ndarray]:
    n = int(dataset_cfg.get("sequence_length", 300))
    dt = float(dataset_cfg.get("dt", 0.1))

    if pose_type == "2d":
        gt = np.zeros((n, 3), dtype=float)
        controls = np.zeros((n, 2), dtype=float)
        controls[:, 0] = 1.0 + 0.2 * np.sin(np.linspace(0.0, 6.0, n))
        controls[:, 1] = 0.12 * np.sin(np.linspace(0.0, 4.0, n))

        for k in range(1, n):
            x, y, yaw = gt[k - 1]
            speed, yaw_rate = controls[k]
            yaw = yaw + yaw_rate * dt
            x = x + speed * np.cos(yaw) * dt
            y = y + speed * np.sin(yaw) * dt
            gt[k] = np.array([x, y, yaw], dtype=float)
        return controls, gt

    gt = np.zeros((n, 6), dtype=float)
    controls = np.zeros((n, 6), dtype=float)
    controls[:, 0] = 0.8 + 0.2 * np.sin(np.linspace(0.0, 5.0, n))
    controls[:, 1] = 0.3 * np.cos(np.linspace(0.0, 3.0, n))
    controls[:, 2] = 0.1 * np.sin(np.linspace(0.0, 2.0, n))
    controls[:, 5] = 0.08 * np.sin(np.linspace(0.0, 4.0, n))

    for k in range(1, n):
        gt[k] = gt[k - 1] + controls[k] * dt

    return controls, gt
