from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.euroc_loader import load_euroc_dataset
from datasets.m2dgr_loader import load_m2dgr_dataset
from datasets.rosbag_loader import load_rosbag_dataset
from utils.csv_dataset import load_dataset_from_csv, save_dataset_to_csv
from utils.generate_gnss import generate_gnss_measurements
from utils.generate_imu import generate_imu_controls

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_dataset(dataset_cfg: dict):
    pose_type = dataset_cfg.get("pose_type", "2d")
    if pose_type == "6d":
        pose_type = "3d"

    mode = dataset_cfg.get("mode", "fused")
    if mode == "gps_only":
        mode = "gnss_only"
        dataset_cfg["mode"] = mode

    _normalize_dataset_paths(dataset_cfg)

    dataset_type = dataset_cfg.get("dataset_type", "synthetic")
    dataset_name = resolve_dataset_name(dataset_cfg, dataset_type)

    generated_csv_path = _resolve_generated_csv_path(dataset_cfg, dataset_name)
    dataset_cfg["generated_csv_path"] = generated_csv_path

    if dataset_type == "euroc":
        pose_type = "3d"
        dataset_cfg["pose_type"] = "6d"
        controls, measurements, gt, dt, timestamps_ns = load_euroc_dataset(dataset_cfg)
    elif dataset_type in ("rosbag", "rosbag1", "rosbag2", "kaist_vio", "kaistvio", "kaist"):
        pose_type = "3d"
        dataset_cfg["pose_type"] = "6d"
        controls, measurements, gt, dt, timestamps_ns = load_rosbag_dataset(dataset_cfg)
    elif dataset_type in ("m2dgr",):
        pose_type = "3d"
        dataset_cfg["pose_type"] = "6d"
        controls, measurements, gt, dt, timestamps_ns = load_m2dgr_dataset(dataset_cfg)
    else:
        controls, gt = generate_imu_controls(dataset_cfg, pose_type=pose_type)
        measurements = generate_gnss_measurements(dataset_cfg, pose_type=pose_type, gt=gt)
        dt = float(dataset_cfg.get("dt", 0.1))
        timestamps_ns = (np.arange(len(gt), dtype=np.int64) * int(round(float(dt) * 1e9))).astype(np.int64)

    csv_path = save_dataset_to_csv(
        generated_csv_path,
        pose_type=pose_type,
        dt=dt,
        controls=controls,
        measurements=measurements,
        gt=gt,
    )

    dataset, gt = load_dataset_from_csv(csv_path, pose_type=pose_type, mode=mode)
    return pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns


def resolve_dataset_name(dataset_cfg: dict, dataset_type: str) -> str:
    configured = str(dataset_cfg.get("dataset_name", "")).strip()
    if configured:
        return configured.lower().replace(" ", "_")

    if dataset_type in ("rosbag", "rosbag1", "rosbag2", "kaist_vio", "kaistvio", "kaist") and "rosbag_path" in dataset_cfg:
        bag_path = Path(dataset_cfg["rosbag_path"])
        if bag_path.name == "metadata.yaml":
            candidate = bag_path.parent.name
        elif bag_path.suffix == ".db3":
            candidate = bag_path.parent.name
        else:
            candidate = bag_path.stem if bag_path.is_file() else bag_path.name
        candidate = candidate.strip()
        if candidate:
            return candidate.lower().replace(" ", "_")

    if dataset_type == "m2dgr" and "m2dgr_bag_path" in dataset_cfg:
        bag_path = Path(dataset_cfg["m2dgr_bag_path"])
        candidate = bag_path.stem if bag_path.is_file() else bag_path.name
        candidate = candidate.strip()
        if candidate:
            return candidate.lower().replace(" ", "_")

    return str(dataset_type).strip().lower().replace(" ", "_") or "dataset"


def _normalize_dataset_paths(dataset_cfg: dict) -> None:
    for key in ("rosbag_path", "m2dgr_bag_path", "m2dgr_gt_txt_path"):
        if key not in dataset_cfg:
            continue
        path = Path(dataset_cfg[key]).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        dataset_cfg[key] = path


def _resolve_generated_csv_path(dataset_cfg: dict, dataset_name: str) -> Path:
    generated_csv_path = dataset_cfg.get("generated_csv_path")
    if generated_csv_path is None:
        return PROJECT_ROOT / "outputs" / f"{dataset_name}_dataset.csv"

    generated_csv_path = Path(generated_csv_path)
    if not generated_csv_path.is_absolute():
        generated_csv_path = PROJECT_ROOT / generated_csv_path

    generic_names = {
        "synthetic_2d.csv",
        "synthetic_3d.csv",
        "synthetic_6d.csv",
        "euroc_6d.csv",
        "kaist_vio_6d.csv",
        "rosbag_6d.csv",
        "m2dgr_6d.csv",
    }
    if generated_csv_path.name in generic_names:
        return generated_csv_path.with_name(f"{dataset_name}_dataset.csv")
    return generated_csv_path
