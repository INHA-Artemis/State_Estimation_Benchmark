from __future__ import annotations

import numpy as np


def diagonal_covariance(values: np.ndarray, minimum: float = 1e-12) -> np.ndarray:
    return np.diag(np.clip(np.asarray(values, dtype=float).reshape(-1), minimum, None))


def kalman_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ innovation
    I_KH = np.eye(P.shape[0], dtype=float) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P, innovation, S, K


def weighted_covariance(values: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cov = np.zeros((values.shape[1], values.shape[1]), dtype=float)
    for idx in range(values.shape[0]):
        residual = values[idx] - mean
        cov += weights[idx] * np.outer(residual, residual)
    return 0.5 * (cov + cov.T)


def cross_variance(
    x_mean: np.ndarray,
    z_mean: np.ndarray,
    sigmas_f: np.ndarray,
    sigmas_h: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]), dtype=float)
    for idx in range(sigmas_f.shape[0]):
        Pxz += weights[idx] * np.outer(sigmas_f[idx] - x_mean, sigmas_h[idx] - z_mean)
    return Pxz


def diagonal_gaussian_logpdf(innovation: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.asarray(variance, dtype=float).reshape(1, -1)
    return -0.5 * (
        np.sum((innovation**2) / variance, axis=1)
        + np.sum(np.log(2.0 * np.pi * variance))
    )
