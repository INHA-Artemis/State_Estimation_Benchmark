from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def align_initialization_with_ground_truth(
    filter_cfg: dict,
    gt: np.ndarray,
    pose_type: str,
    mode: str,
) -> None:
    """Anchor the initial filter mean to the first GT pose for imu-only runs."""
    if str(mode).lower() != "imu_only":
        return
    if gt is None or len(gt) == 0:
        return

    init_cfg = filter_cfg.setdefault("initialization", {})
    first_gt = np.asarray(gt[0], dtype=float).reshape(-1)

    if pose_type == "2d":
        mean = _fit_pose(first_gt, 3)
    else:
        mean = _fit_3d_state_from_pose(first_gt)

    init_cfg["mean"] = mean.tolist()


def _fit_pose(values: Iterable[float], dim: int) -> np.ndarray:
    vector = np.asarray(list(values), dtype=float).reshape(-1)
    out = np.zeros(dim, dtype=float)
    out[: min(dim, vector.size)] = vector[: min(dim, vector.size)]
    return out


def _fit_3d_state_from_pose(values: Iterable[float]) -> np.ndarray:
    """Map GT pose [p, rpy] to state [p, v, orientation, imu_bias]."""
    vector = np.asarray(list(values), dtype=float).reshape(-1)
    out = np.zeros(15, dtype=float)

    if vector.size >= 3:
        out[0:3] = vector[0:3]

    if vector.size >= 9:
        out[6:9] = vector[6:9]
    elif vector.size >= 6:
        out[6:9] = vector[3:6]

    return out
