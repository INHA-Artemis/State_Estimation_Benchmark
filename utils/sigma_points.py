from __future__ import annotations

import numpy as np


def merwe_sigma_points(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    dim = mean.size
    lambda_ = float(alpha) ** 2 * (dim + float(kappa)) - dim
    scale = dim + lambda_
    if scale <= 0:
        raise ValueError("UKF sigma point scale must be positive. Check alpha/beta/kappa.")

    chol = robust_cholesky(scale * cov)
    sigmas = np.zeros((2 * dim + 1, dim), dtype=float)
    sigmas[0] = mean
    for i in range(dim):
        sigmas[i + 1] = mean + chol[:, i]
        sigmas[dim + i + 1] = mean - chol[:, i]

    Wm = np.full(2 * dim + 1, 1.0 / (2.0 * scale), dtype=float)
    Wc = np.full(2 * dim + 1, 1.0 / (2.0 * scale), dtype=float)
    Wm[0] = lambda_ / scale
    Wc[0] = lambda_ / scale + (1.0 - float(alpha) ** 2 + float(beta))
    return sigmas, Wm, Wc, lambda_


def robust_cholesky(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eye = np.eye(matrix.shape[0], dtype=float)
    jitter = 1e-9
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    eigvals, eigvecs = np.linalg.eigh(matrix)
    spd = eigvecs @ np.diag(np.clip(eigvals, 1e-9, None)) @ eigvecs.T
    return np.linalg.cholesky(0.5 * (spd + spd.T) + jitter * eye)
