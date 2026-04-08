from __future__ import annotations

from typing import Iterable, Literal

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    axis = phi / theta
    k_matrix = skew(axis)
    return np.eye(3) + np.sin(theta) * k_matrix + (1.0 - np.cos(theta)) * (k_matrix @ k_matrix)


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def nearest_spd(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (spd + spd.T)


def fit_vector(values: Iterable[float], dim: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == dim:
        return vector
    out = np.zeros(dim, dtype=float)
    out[: min(dim, vector.size)] = vector[: min(dim, vector.size)]
    return out


def fit_diag(
    values: Iterable[float],
    dim: int,
    fill_missing: Literal["last", "zero"] = "last",
) -> np.ndarray:
    diag = np.asarray(values, dtype=float).reshape(-1)
    if diag.size == dim:
        return diag
    if diag.size == 0:
        return np.zeros(dim, dtype=float)
    if diag.size == 1:
        return np.full(dim, float(diag.item()), dtype=float)

    out = np.zeros(dim, dtype=float)
    out[: min(dim, diag.size)] = diag[: min(dim, diag.size)]
    if diag.size < dim and fill_missing == "last":
        out[diag.size :] = diag[-1]
    return out


def compute_rmse(estimates: np.ndarray, gt: np.ndarray, pose_type: str) -> float:
    pos_dim = 2 if pose_type == "2d" else 3
    err = estimates[:, :pos_dim] - gt[:, :pos_dim]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
