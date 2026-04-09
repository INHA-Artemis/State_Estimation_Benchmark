from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.estimated_kalman_filter import ExtendedKalmanFilter
from utils.filter_config import (
    load_visualization_config,
    merge_visualization_config,
    normalize_position_filter_config_for_pose,
)
from utils.filter_initialization import align_initialization_with_ground_truth
from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.save_estimates import save_estimates_to_csv
from utils.visualization import plot_position_error_norm, plot_results, save_trajectory_animation
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run Extended Kalman Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--ekf-config", default=str(PROJECT_ROOT / "config" / "ekf.yaml"))
    parser.add_argument("--visualization-config", default=str(PROJECT_ROOT / "config" / "output_visualization.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    ekf_cfg = load_yaml(Path(args.ekf_config))
    ekf_cfg["visualization"] = merge_visualization_config(
        load_visualization_config(args.visualization_config),
        ekf_cfg.get("visualization"),
    )

    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)
    normalize_position_filter_config_for_pose(ekf_cfg, pose_type)
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


if __name__ == "__main__":
    main()
