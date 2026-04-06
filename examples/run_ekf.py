from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.euroc_loader import load_euroc_dataset
from datasets.m2dgr_loader import load_m2dgr_dataset
from datasets.rosbag_loader import load_rosbag_dataset
from filters.estimated_kalman_filter import ExtendedKalmanFilter
from utils.csv_dataset import load_dataset_from_csv, save_dataset_to_csv
from utils.filter_initialization import align_initialization_with_ground_truth
from utils.generate_gnss import generate_gnss_measurements
from utils.generate_imu import generate_imu_controls
from utils.math_utils import compute_rmse
from utils.save_estimates import save_estimates_to_csv
from utils.visualization import plot_position_error_norm, plot_results, save_trajectory_animation
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run Extended Kalman Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--ekf-config", default=str(PROJECT_ROOT / "config" / "ekf.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    ekf_cfg = load_yaml(Path(args.ekf_config))

    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = _prepare_dataset(dataset_cfg)
    _normalize_filter_config_for_pose(ekf_cfg, pose_type)
    align_initialization_with_ground_truth(ekf_cfg, gt, pose_type, dataset_cfg.get("mode", "fused"))

    if np.isscalar(dt):
        dt_values = np.full(len(dataset), float(dt), dtype=float)
    else:
        dt_values = np.asarray(dt, dtype=float).reshape(-1)

    ekf = ExtendedKalmanFilter.from_configs(dataset_cfg, ekf_cfg)

    ekf_start = time.perf_counter()
    estimates = ekf.run(dataset)
    ekf_runtime = time.perf_counter() - ekf_start

    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[EKF] Pose type: {pose_type}")
    print(f"[EKF] Dataset CSV: {csv_path}")
    print(f"[EKF] Steps: {len(dataset)}")
    print(f"[EKF] RMSE (position): {rmse:.4f}")
    print(f"[EKF] Runtime (filter only): {ekf_runtime:.3f} sec")

    out_dir = Path(args.output_dir)
    estimates_csv_path = out_dir / f"{dataset_name}_ekf_estimates.csv"
    saved_estimates_csv = save_estimates_to_csv(
        estimates_csv_path,
        estimates,
        pose_type,
        timestamps_ns=timestamps_ns,
        dt_values=dt_values,
    )
    print(f"[EKF] Estimates CSV saved: {saved_estimates_csv}")

    vis_cfg = dict(ekf_cfg.get("visualization", {}))
    vis_cfg["estimator_label"] = "EKF"

    plot_path = out_dir / f"{dataset_name}_ekf_trajectory.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path, visual_cfg=vis_cfg)
    print(f"[EKF] Plot saved: {plot_path}")

    error_plot_path = out_dir / f"{dataset_name}_ekf_position_error_norm.png"
    plot_position_error_norm(estimates, gt, pose_type=pose_type, save_path=error_plot_path, visual_cfg=vis_cfg)
    print(f"[EKF] Error plot saved: {error_plot_path}")

    anim_cfg = vis_cfg.get("animation", {})
    if vis_cfg.get("save_animation", False):
        anim_format = str(anim_cfg.get("format", "mp4")).lower()
        if anim_format not in {"mp4", "gif"}:
            raise ValueError("visualization.animation.format must be 'mp4' or 'gif'.")
        video_path = out_dir / f"{dataset_name}_ekf_trajectory.{anim_format}"
        saved = save_trajectory_animation(
            estimates,
            gt,
            pose_type=pose_type,
            save_path=video_path,
            fps=int(anim_cfg.get("fps", 20)),
            tail_length=int(anim_cfg.get("tail_length", 80)),
            moving_average_window=int(vis_cfg.get("error_moving_average_window", 50)),
            estimator_label="EKF",
        )
        if saved:
            print(f"[EKF] Animation saved: {video_path}")
        else:
            writer_name = "ffmpeg" if anim_format == "mp4" else "pillow"
            print(f"[EKF] Animation skipped: {writer_name} writer is not available.")

    total_runtime = time.perf_counter() - total_start
    print(f"[EKF] Runtime (total): {total_runtime:.3f} sec")


def _normalize_filter_config_for_pose(filter_cfg: dict, pose_type: str) -> None:
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


def _prepare_dataset(dataset_cfg: dict):
    pose_type = dataset_cfg.get("pose_type", "2d")
    if pose_type == "6d":
        pose_type = "3d"

    mode = dataset_cfg.get("mode", "fused")
    if mode == "gps_only":
        mode = "gnss_only"
        dataset_cfg["mode"] = mode

    if "rosbag_path" in dataset_cfg:
        bag_path = Path(dataset_cfg["rosbag_path"]).expanduser()
        if not bag_path.is_absolute():
            bag_path = PROJECT_ROOT / bag_path
        dataset_cfg["rosbag_path"] = bag_path

    if "m2dgr_bag_path" in dataset_cfg:
        bag_path = Path(dataset_cfg["m2dgr_bag_path"]).expanduser()
        if not bag_path.is_absolute():
            bag_path = PROJECT_ROOT / bag_path
        dataset_cfg["m2dgr_bag_path"] = bag_path

    if "m2dgr_gt_txt_path" in dataset_cfg:
        gt_path = Path(dataset_cfg["m2dgr_gt_txt_path"]).expanduser()
        if not gt_path.is_absolute():
            gt_path = PROJECT_ROOT / gt_path
        dataset_cfg["m2dgr_gt_txt_path"] = gt_path

    dataset_type = dataset_cfg.get("dataset_type", "synthetic")
    dataset_name = _resolve_dataset_name(dataset_cfg, dataset_type)

    generated_csv_path = dataset_cfg.get("generated_csv_path")
    if generated_csv_path is None:
        generated_csv_path = PROJECT_ROOT / "outputs" / f"{dataset_name}_dataset.csv"
    else:
        generated_csv_path = Path(generated_csv_path)
        if not generated_csv_path.is_absolute():
            generated_csv_path = PROJECT_ROOT / generated_csv_path
        generic_names = {"synthetic_2d.csv", "synthetic_3d.csv", "synthetic_6d.csv", "euroc_6d.csv", "kaist_vio_6d.csv", "rosbag_6d.csv", "m2dgr_6d.csv"}
        if generated_csv_path.name in generic_names:
            generated_csv_path = generated_csv_path.with_name(f"{dataset_name}_dataset.csv")
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


def _resolve_dataset_name(dataset_cfg: dict, dataset_type: str) -> str:
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


if __name__ == "__main__":
    main()
