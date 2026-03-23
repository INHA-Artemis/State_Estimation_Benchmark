from __future__ import annotations

import numpy as np


def generate_gnss_measurements(dataset_cfg: dict, pose_type: str, gt: np.ndarray) -> np.ndarray:
    seed = int(dataset_cfg.get("seed", 10))
    rng = np.random.default_rng(seed)

    default_std = [0.7, 0.7] if pose_type == "2d" else [0.7, 0.7, 0.7]
    meas_std = np.array(
        dataset_cfg.get("gnss_noise_std", dataset_cfg.get("gps_noise_std", default_std)),
        dtype=float,
    )

    measurement_dim = 2 if pose_type == "2d" else 3
    if measurement_dim == 3 and meas_std.size == 2:
        meas_std = np.array([meas_std[0], meas_std[1], 0.7], dtype=float)

    return gt[:, :measurement_dim] + rng.normal(0.0, meas_std[:measurement_dim], size=(len(gt), measurement_dim))
