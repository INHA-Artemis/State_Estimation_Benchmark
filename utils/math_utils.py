import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    axis = phi / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)



