from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from models.state_model import state_dim
from utils.yaml_loader import load_yaml


def load_visualization_config(path: str | Path) -> dict:
    cfg = load_yaml(Path(path))
    return dict(cfg.get("visualization", {}))


def merge_visualization_config(common_cfg: dict | None, filter_cfg: dict | None) -> dict:
    merged = deepcopy(common_cfg or {})
    _deep_update(merged, filter_cfg or {})
    return merged


def normalize_position_filter_config_for_pose(filter_cfg: dict, pose_type: str) -> None:
    if pose_type == "6d":
        pose_type = "3d"

    measurement_cfg = filter_cfg.setdefault("measurement_model", {})
    evaluation_cfg = filter_cfg.setdefault("evaluation", {})
    visual_cfg = filter_cfg.setdefault("visualization", {})
    init_cfg = filter_cfg.setdefault("initialization", {})
    motion_cfg = filter_cfg.setdefault("motion_model", {})

    if pose_type != "3d":
        measurement_cfg["position_indices"] = [0, 1]
        measurement_cfg["measurement_noise_diag"] = _resize_list(
            measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]),
            2,
            default_fill=0.7,
        )
        evaluation_cfg["position_indices"] = [0, 1]
        evaluation_cfg["yaw_index"] = 2
        visual_cfg["position_indices"] = [0, 1]

        dim_2d = state_dim("2d")
        init_cfg["mean"] = _resize_list(init_cfg.get("mean", [0.0, 0.0, 0.0]), dim_2d, default_fill=0.0)
        init_cfg["cov_diag"] = _resize_list(init_cfg.get("cov_diag", [1.0, 1.0, 0.3]), dim_2d, default_fill=0.3)
        motion_cfg["process_noise_diag"] = _resize_list(
            motion_cfg.get("process_noise_diag", [0.02, 0.02, 0.005]),
            dim_2d,
            default_fill=0.005,
        )
        return

    # Position measurement/evaluation/visualization are always 3D indices.
    measurement_cfg["position_indices"] = [0, 1, 2]
    measurement_cfg["measurement_noise_diag"] = _resize_list(
        measurement_cfg.get("measurement_noise_diag", [0.05, 0.05, 0.05]),
        3,
        default_fill=0.05,
    )

    evaluation_cfg["position_indices"] = [0, 1, 2]
    evaluation_cfg["yaw_index"] = 5
    visual_cfg["position_indices"] = [0, 1, 2]

    # For EKF/UKF/PF 3D state: [p(3), v(3), orientation(3), imu_bias(6)]
    dim_3d = state_dim("3d")

    default_mean = np.zeros(dim_3d, dtype=float)
    default_cov = np.array(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02],
        dtype=float,
    )
    default_process = np.array(
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002],
        dtype=float,
    )

    init_cfg["mean"] = _resize_list(init_cfg.get("mean", default_mean), dim_3d, default_fill=0.0)
    init_cfg["cov_diag"] = _resize_list(init_cfg.get("cov_diag", default_cov), dim_3d, default_fill=float(default_cov[-1]))
    motion_cfg["process_noise_diag"] = _resize_list(
        motion_cfg.get("process_noise_diag", default_process),
        dim_3d,
        default_fill=float(default_process[-1]),
    )

    motion_cfg.setdefault("linear_input_type", "velocity")
    motion_cfg.setdefault("gravity", [0.0, 0.0, -9.81])


def _resize_list(values, dim: int, default_fill: float) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == dim:
        return arr.tolist()
    if arr.size == 0:
        return np.full(dim, float(default_fill), dtype=float).tolist()
    if arr.size == 1:
        return np.full(dim, float(arr.item()), dtype=float).tolist()

    out = np.full(dim, float(default_fill), dtype=float)
    out[: min(dim, arr.size)] = arr[: min(dim, arr.size)]
    if arr.size < dim:
        out[arr.size :] = arr[-1]
    return out.tolist()


def _deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
            continue
        base[key] = deepcopy(value)
