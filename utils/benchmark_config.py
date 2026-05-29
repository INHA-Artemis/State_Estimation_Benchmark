from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


KAIST_VIO_DEFAULTS = {
    "dataset_type": "kaist_vio",
    "bag": "datasets/KAIST_VIO/infinite.bag",
    "dataset_name": "kaist_vio_infinite",
    "imu_topic": "/mavros/imu/data",
    "gt_topic": "/pose_transformed",
    "linear_source": "accel",
}

M2DGR_DEFAULTS = {
    "dataset_type": "m2dgr",
    "bag": "datasets/M2DGR/street_02.bag",
    "gt_txt": "datasets/M2DGR/street_02.txt",
    "dataset_name": "m2dgr_street_02",
    "imu_topic": "/handsfree/imu",
    "gnss_topic": "/ublox/fix",
    "linear_source": "gt_velocity",
}


def apply_kaist_vio_dataset_config(
    args: argparse.Namespace,
    compare_cfg: dict[str, Any],
    project_root: Path,
) -> None:
    apply_benchmark_dataset_config(args, compare_cfg, project_root)


def apply_benchmark_dataset_config(
    args: argparse.Namespace,
    compare_cfg: dict[str, Any],
    project_root: Path,
) -> None:
    dataset_cfg = _dataset_section(compare_cfg)
    configured_type = _normalize_dataset_type(
        _dataset_value(dataset_cfg, "dataset_type", "type", default=KAIST_VIO_DEFAULTS["dataset_type"])
    )
    requested_type = getattr(args, "dataset_type", None)
    dataset_type = _normalize_dataset_type(
        requested_type or configured_type
    )
    if requested_type is not None and dataset_type != configured_type:
        dataset_cfg = {}
    defaults = M2DGR_DEFAULTS if dataset_type == "m2dgr" else KAIST_VIO_DEFAULTS
    values = {
        "dataset_type": dataset_type,
        "bag": _dataset_value(dataset_cfg, "bag", "rosbag_path", "m2dgr_bag_path", default=defaults["bag"]),
        "dataset_name": _dataset_value(dataset_cfg, "dataset_name", "name", default=defaults["dataset_name"]),
        "imu_topic": _dataset_value(
            dataset_cfg,
            "imu_topic",
            "rosbag_imu_topic",
            "m2dgr_imu_topic",
            default=defaults["imu_topic"],
        ),
        "linear_source": _dataset_value(
            dataset_cfg,
            "linear_source",
            "rosbag_linear_source",
            "m2dgr_linear_source",
            default=defaults["linear_source"],
        ),
    }
    if dataset_type == "m2dgr":
        values["gt_txt"] = _dataset_value(
            dataset_cfg,
            "gt_txt",
            "gt_path",
            "m2dgr_gt_txt_path",
            default=M2DGR_DEFAULTS["gt_txt"],
        )
        values["gnss_topic"] = _dataset_value(
            dataset_cfg,
            "gnss_topic",
            "m2dgr_gnss_topic",
            default=M2DGR_DEFAULTS["gnss_topic"],
        )
    else:
        values["gt_topic"] = _dataset_value(
            dataset_cfg,
            "gt_topic",
            "rosbag_gt_topic",
            default=KAIST_VIO_DEFAULTS["gt_topic"],
        )

    for key, value in values.items():
        if getattr(args, key, None) is not None:
            continue
        if key in ("bag", "gt_txt"):
            value = _resolve_project_path(str(value), project_root)
        setattr(args, key, value)


def default_benchmark_output_root(project_root: Path, benchmark_name: str, dataset_type: str) -> Path:
    return project_root / "outputs" / "benchmarks" / benchmark_name / dataset_family_slug(dataset_type)


def dataset_family_slug(dataset_type: str) -> str:
    return "m2dgr" if _normalize_dataset_type(dataset_type) == "m2dgr" else "kaist_vio"


def dataset_run_slug(bag_path: str, dataset_name: str, dataset_type: str) -> str:
    if _normalize_dataset_type(dataset_type) == "m2dgr":
        stem = Path(bag_path).stem
        return _slug(stem.replace("_", "")) or _slug(str(dataset_name).replace("m2dgr_", "")) or "run"

    stem = Path(bag_path).stem
    if stem:
        return _slug(stem)

    name = str(dataset_name)
    prefix = "kaist_vio_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return _slug(name) or "run"


def _dataset_section(compare_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = compare_cfg.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return {}

    dataset_type = str(dataset_cfg.get("type", dataset_cfg.get("dataset_type", ""))).strip().lower()
    nested_key = _normalize_dataset_type(dataset_type)
    nested = dataset_cfg.get(nested_key)
    if isinstance(nested, dict):
        return nested
    if dataset_type == "kaist":
        nested = dataset_cfg.get("kaist_vio")
        if isinstance(nested, dict):
            return nested
    if dataset_type in ("", "kaist_vio", "kaistvio", "kaist", "rosbag", "rosbag1", "rosbag2", "m2dgr"):
        return dataset_cfg
    return {}


def _dataset_value(dataset_cfg: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = dataset_cfg.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _resolve_project_path(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def _normalize_dataset_type(value: Any) -> str:
    dataset_type = str(value).strip().lower()
    if dataset_type in ("m2dgr",):
        return "m2dgr"
    return "kaist_vio"


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
