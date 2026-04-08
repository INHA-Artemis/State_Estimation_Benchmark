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
    meas_std = _fit_std(meas_std, measurement_dim, default_fill=0.7)

    noise_model = str(dataset_cfg.get("gnss_noise_model", "gaussian")).strip().lower()
    noise = rng.normal(0.0, meas_std, size=(len(gt), measurement_dim))

    if noise_model == "outlier_mixture":
        outlier_prob = float(dataset_cfg.get("gnss_outlier_prob", 0.05))
        outlier_prob = float(np.clip(outlier_prob, 0.0, 1.0))
        outlier_std = _fit_std(
            np.asarray(dataset_cfg.get("gnss_outlier_std", meas_std * 20.0), dtype=float),
            measurement_dim,
            default_fill=float(meas_std[-1]) * 20.0,
        )
        outlier_mask = rng.random(len(gt)) < outlier_prob
        outlier_count = int(np.count_nonzero(outlier_mask))
        if outlier_count:
            noise[outlier_mask] = rng.normal(0.0, outlier_std, size=(outlier_count, measurement_dim))
    elif noise_model != "gaussian":
        raise ValueError("gnss_noise_model must be 'gaussian' or 'outlier_mixture'.")

    return gt[:, :measurement_dim] + noise


def _fit_std(values: np.ndarray, dim: int, default_fill: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.full(dim, float(default_fill), dtype=float)
    if values.size >= dim:
        return values[:dim].astype(float)
    out = np.full(dim, float(default_fill), dtype=float)
    out[: values.size] = values
    if values.size > 0:
        out[values.size :] = values[-1]
    return out
