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
        mean = _fit_pose(first_gt, 6)

    init_cfg["mean"] = mean.tolist()


def _fit_pose(values: Iterable[float], dim: int) -> np.ndarray:
    vector = np.asarray(list(values), dtype=float).reshape(-1)
    out = np.zeros(dim, dtype=float)
    out[: min(dim, vector.size)] = vector[: min(dim, vector.size)]
    return out
