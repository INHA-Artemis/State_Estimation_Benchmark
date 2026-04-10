from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.filter_config import load_visualization_config, merge_visualization_config
from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.save_estimates import save_estimates_to_csv
from utils.visualization import (
    control_trajectory_from_dataset,
    measurement_trajectory_from_dataset,
    plot_position_error_norm,
    plot_results,
    save_trajectory_animation,
)
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run no-filter baseline and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--visualization-config", default=str(PROJECT_ROOT / "config" / "output_visualization.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    vis_cfg = merge_visualization_config(load_visualization_config(args.visualization_config), {})

    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)

    if np.isscalar(dt):
        dt_values = np.full(len(dataset), float(dt), dtype=float)
    else:
        dt_values = np.asarray(dt, dtype=float).reshape(-1)

    baseline_start = time.perf_counter()
    estimates = _estimate_without_filter(dataset, gt, pose_type, str(dataset_cfg.get("mode", "fused")).lower())
    baseline_runtime = time.perf_counter() - baseline_start

    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[NoFilter] Pose type: {pose_type}")
    print(f"[NoFilter] Dataset CSV: {csv_path}")
    print(f"[NoFilter] Steps: {len(dataset)}")
    print(f"[NoFilter] RMSE (position): {rmse:.4f}")
    print(f"[NoFilter] Runtime (filter only): {baseline_runtime:.3f} sec")

    out_dir = Path(args.output_dir)
    estimates_csv_path = out_dir / f"{dataset_name}_nofilter_estimates.csv"
    saved_estimates_csv = save_estimates_to_csv(
        estimates_csv_path,
        estimates,
        pose_type,
        timestamps_ns=timestamps_ns,
        dt_values=dt_values,
    )
    print(f"[NoFilter] Estimates CSV saved: {saved_estimates_csv}")

    vis_cfg = dict(vis_cfg)
    vis_cfg["estimator_label"] = "NoFilter"

    plot_path = out_dir / f"{dataset_name}_nofilter_trajectory.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path, visual_cfg=vis_cfg)
    print(f"[NoFilter] Plot saved: {plot_path}")

    error_plot_path = out_dir / f"{dataset_name}_nofilter_position_error_norm.png"
    plot_position_error_norm(estimates, gt, pose_type=pose_type, save_path=error_plot_path, visual_cfg=vis_cfg)
    print(f"[NoFilter] Error plot saved: {error_plot_path}")

    anim_cfg = vis_cfg.get("animation", {})
    if vis_cfg.get("save_animation", False):
        anim_format = str(anim_cfg.get("format", "mp4")).lower()
        if anim_format not in {"mp4", "gif"}:
            raise ValueError("visualization.animation.format must be 'mp4' or 'gif'.")
        video_path = out_dir / f"{dataset_name}_nofilter_trajectory.{anim_format}"
        saved = save_trajectory_animation(
            estimates,
            gt,
            pose_type=pose_type,
            save_path=video_path,
            fps=int(anim_cfg.get("fps", 20)),
            tail_length=int(anim_cfg.get("tail_length", 80)),
            moving_average_window=int(vis_cfg.get("error_moving_average_window", 50)),
            estimator_label="NoFilter",
        )
        if saved:
            print(f"[NoFilter] Animation saved: {video_path}")
        else:
            writer_name = "ffmpeg" if anim_format == "mp4" else "pillow"
            print(f"[NoFilter] Animation skipped: {writer_name} writer is not available.")

    total_runtime = time.perf_counter() - total_start
    print(f"[NoFilter] Runtime (total): {total_runtime:.3f} sec")


def _estimate_without_filter(dataset: list[dict], gt: np.ndarray, pose_type: str, mode: str) -> np.ndarray:
    measurement_traj = measurement_trajectory_from_dataset(dataset, gt, pose_type)
    control_traj = control_trajectory_from_dataset(dataset, gt, pose_type)

    if mode == "imu_only":
        if control_traj is None:
            raise ValueError("NoFilter imu_only mode requires control data.")
        return control_traj

    if mode == "gnss_only":
        if measurement_traj is None:
            raise ValueError("NoFilter gnss_only mode requires measurement data.")
        return measurement_traj

    # fused: no probabilistic fusion; prefer direct GNSS baseline when available.
    if measurement_traj is not None:
        return measurement_traj
    if control_traj is not None:
        return control_traj

    raise ValueError("NoFilter fused mode requires at least one of control or measurement data.")


if __name__ == "__main__":
    main()
