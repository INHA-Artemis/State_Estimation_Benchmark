from __future__ import annotations

import numpy as np

from utils.math_utils import wrap_angle
from utils.rotation_utils import rot_to_rpy, rpy_to_rot


def as_matrix(Rot: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    X = np.eye(5, dtype=float)
    X[:3, :3] = Rot
    X[:3, 3] = v
    X[:3, 4] = p
    return X


def from_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(X[:3, :3], dtype=float),
        np.asarray(X[:3, 3], dtype=float).reshape(3),
        np.asarray(X[:3, 4], dtype=float).reshape(3),
    )


def pose_to_state(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(mean, dtype=float).reshape(6)
    return pose[0:3].copy(), rpy_to_rot(pose[3:6])


def pose_from_state(Rot: np.ndarray, p: np.ndarray) -> np.ndarray:
    rpy = np.array([wrap_angle(angle) for angle in rot_to_rpy(Rot)], dtype=float)
    return np.concatenate([np.asarray(p, dtype=float).reshape(3), rpy])


def propagate_mean(
    Rot: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    control: np.ndarray,
    dt: float,
    *,
    use_imu_velocity: bool,
    use_imu_rotation: bool,
    translation_input_frame: str,
    translation_input_type: str,
    rotation_input_type: str,
    rotation_representation: str,
    gravity: np.ndarray,
    velocity_blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rot = np.asarray(Rot, dtype=float).reshape(3, 3)
    v = np.asarray(v, dtype=float).reshape(3).copy()
    p = np.asarray(p, dtype=float).reshape(3).copy()
    control = np.asarray(control, dtype=float).reshape(-1)
    dt = float(dt)

    if use_imu_rotation:
        rot_vec = control[3:6]
        if rotation_input_type == "rate":
            rot_vec = rot_vec * dt
        elif rotation_input_type != "increment":
            raise ValueError(f"Unsupported rotation_input_type: {rotation_input_type}")

        if rotation_representation == "rotvec":
            Rot = Rot @ so3_exp(rot_vec)
        elif rotation_representation == "euler":
            Rot = rpy_to_rot(rot_to_rpy(Rot) + rot_vec)
        else:
            raise ValueError(f"Unsupported rotation_representation: {rotation_representation}")

    if not use_imu_velocity:
        return Rot, v, p + v * dt

    trans = control[0:3]
    if translation_input_frame == "body":
        trans_world = Rot @ trans
    elif translation_input_frame == "world":
        trans_world = trans
    else:
        raise ValueError(f"Unsupported translation_input_frame: {translation_input_frame}")

    if translation_input_type == "velocity":
        v = velocity_blend * v + (1.0 - velocity_blend) * trans_world
        p = p + v * dt
    elif translation_input_type == "acceleration":
        accel_world = trans_world + np.asarray(gravity, dtype=float).reshape(3)
        p = p + v * dt + 0.5 * accel_world * dt**2
        v = v + accel_world * dt
    elif translation_input_type == "increment":
        p = p + trans_world
        if dt > 1e-12:
            inferred_v = trans_world / dt
            v = velocity_blend * v + (1.0 - velocity_blend) * inferred_v
    else:
        raise ValueError(f"Unsupported translation_input_type: {translation_input_type}")
    return Rot, v, p


def error_transition(dt: float) -> np.ndarray:
    Phi = np.eye(9, dtype=float)
    Phi[6:9, 3:6] = np.eye(3, dtype=float) * float(dt)
    return Phi


def position_measurement_matrix(p: np.ndarray, indices: np.ndarray) -> np.ndarray:
    H = np.zeros((len(indices), 9), dtype=float)
    for row, idx in enumerate(np.asarray(indices, dtype=int)):
        H[row, 0:3] = -skew(p)[idx, :]
        H[row, 6 + idx] = 1.0
    return H


def se23_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(9)
    R = so3_exp(xi[0:3])
    J = so3_left_jacobian(xi[0:3])
    X = np.eye(5, dtype=float)
    X[:3, :3] = R
    X[:3, 3] = J @ xi[3:6]
    X[:3, 4] = J @ xi[6:9]
    return X


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / theta**2) * K @ K


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * K @ K
    return np.eye(3) + ((1.0 - np.cos(theta)) / theta**2) * K + ((theta - np.sin(theta)) / theta**3) * K @ K
