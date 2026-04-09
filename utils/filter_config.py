from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from utils.yaml_loader import load_yaml


def load_visualization_config(path: str | Path) -> dict:
    cfg = load_yaml(Path(path))
    return dict(cfg.get("visualization", {}))


def merge_visualization_config(common_cfg: dict | None, filter_cfg: dict | None) -> dict:
    merged = deepcopy(common_cfg or {})
    _deep_update(merged, filter_cfg or {})
    return merged


def normalize_position_filter_config_for_pose(filter_cfg: dict, pose_type: str) -> None:
    if pose_type != "3d":
        return

    measurement_cfg = filter_cfg.setdefault("measurement_model", {})
    if len(list(measurement_cfg.get("position_indices", [0, 1]))) < 3:
        measurement_cfg["position_indices"] = [0, 1, 2]

    measurement_noise = list(measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]))
    if len(measurement_noise) == 1:
        measurement_noise = measurement_noise * 3
    elif len(measurement_noise) == 2:
        measurement_noise.append(measurement_noise[-1])
    elif len(measurement_noise) > 3:
        measurement_noise = measurement_noise[:3]
    measurement_cfg["measurement_noise_diag"] = measurement_noise

    evaluation_cfg = filter_cfg.setdefault("evaluation", {})
    if len(list(evaluation_cfg.get("position_indices", [0, 1]))) < 3:
        evaluation_cfg["position_indices"] = [0, 1, 2]

    visual_cfg = filter_cfg.setdefault("visualization", {})
    if len(list(visual_cfg.get("position_indices", [0, 1]))) < 3:
        visual_cfg["position_indices"] = [0, 1, 2]


def _deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
            continue
        base[key] = deepcopy(value)
