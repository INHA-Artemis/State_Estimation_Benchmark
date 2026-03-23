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

def compute_rmse(estimates: np.ndarray, gt: np.ndarray, pose_type: str) -> float:
    pos_dim = 2 if pose_type == "2d" else 3
    err = estimates[:, :pos_dim] - gt[:, :pos_dim]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
