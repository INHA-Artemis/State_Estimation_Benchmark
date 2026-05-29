from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.math_utils import fit_vector


def normalize_pose_type(pose_type: str) -> str:
    if pose_type == "6d":
        pose_type = "3d"
    if pose_type not in {"2d", "3d"}:
        raise ValueError("pose_type must be '2d' or '3d'.")
    return pose_type


def position_dim(pose_type: str) -> int:
    return 2 if normalize_pose_type(pose_type) == "2d" else 3


def state_dim(pose_type: str) -> int:
    return 2 * position_dim(pose_type)


def fit_state(mean: Iterable[float] | None, pose_type: str) -> np.ndarray:
    dim = state_dim(pose_type)
    if mean is None:
        return np.zeros(dim, dtype=float)
    values = np.asarray(mean, dtype=float).reshape(-1)
    if normalize_pose_type(pose_type) == "3d" and values.size >= 6:
        return np.concatenate([values[0:3], values[3:6]])
    if normalize_pose_type(pose_type) == "2d" and values.size >= 4:
        return np.array([values[0], values[1], values[2], values[3]], dtype=float)
    return fit_vector(values, dim)


def transition_matrix(pos_dim: int, dt: float) -> np.ndarray:
    dim = 2 * int(pos_dim)
    F = np.eye(dim, dtype=float)
    F[:pos_dim, pos_dim:] = np.eye(pos_dim, dtype=float) * float(dt)
    return F


def control_vector(
    control: Iterable[float] | None,
    pos_dim: int,
    dt: float,
    control_input_type: str,
    accel_bias: np.ndarray,
    gravity: np.ndarray,
) -> np.ndarray:
    b = np.zeros(2 * int(pos_dim), dtype=float)
    if str(control_input_type).lower() != "acceleration" or control is None:
        return b
    u = fit_vector(np.asarray(control, dtype=float).reshape(-1), pos_dim)
    accel = u - fit_vector(accel_bias, pos_dim) + fit_vector(gravity, pos_dim)
    dt = float(dt)
    b[:pos_dim] = 0.5 * accel * dt**2
    b[pos_dim:] = accel * dt
    return b


def transition_function(
    x: np.ndarray,
    control: Iterable[float] | None,
    pos_dim: int,
    dt: float,
    control_input_type: str,
    accel_bias: np.ndarray,
    gravity: np.ndarray,
) -> np.ndarray:
    return transition_matrix(pos_dim, dt) @ x + control_vector(
        control,
        pos_dim,
        dt,
        control_input_type,
        accel_bias,
        gravity,
    )


def measurement_matrix(indices: np.ndarray, dim: int) -> np.ndarray:
    H = np.zeros((len(indices), dim), dtype=float)
    for row, idx in enumerate(np.asarray(indices, dtype=int)):
        H[row, idx] = 1.0
    return H


def measurement_function(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)[np.asarray(indices, dtype=int)]


def validate_measurement(measurement: Iterable[float], size: int) -> np.ndarray:
    z = np.asarray(measurement, dtype=float).reshape(-1)
    if z.size != int(size):
        raise ValueError("measurement size must match measurement indices.")
    return z


def pose_from_state(x: np.ndarray, pose_type: str) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if normalize_pose_type(pose_type) == "2d":
        return np.array([x[0], x[1], 0.0], dtype=float)
    return np.array([x[0], x[1], x[2], 0.0, 0.0, 0.0], dtype=float)
