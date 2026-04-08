from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.invariant_kalman_filter import InvariantKalmanFilter
from utils.filter_initialization import align_initialization_with_ground_truth
from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.save_estimates import save_estimates_to_csv
from utils.visualization import plot_position_error_norm, plot_results, save_trajectory_animation
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run Invariant Kalman Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--inekf-config", default=str(PROJECT_ROOT / "config" / "inekf.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    inekf_cfg = load_yaml(Path(args.inekf_config))

    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)
    _normalize_filter_config_for_pose(inekf_cfg, pose_type)
    align_initialization_with_ground_truth(inekf_cfg, gt, pose_type, dataset_cfg.get("mode", "fused"))

    if np.isscalar(dt):
        dt_values = np.full(len(dataset), float(dt), dtype=float)
    else:
        dt_values = np.asarray(dt, dtype=float).reshape(-1)

    inekf = InvariantKalmanFilter.from_configs(dataset_cfg, inekf_cfg)

    inekf_start = time.perf_counter()
    estimates = inekf.run(dataset)
    inekf_runtime = time.perf_counter() - inekf_start

    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[InEKF] Pose type: {pose_type}")
    print(f"[InEKF] Dataset CSV: {csv_path}")
    print(f"[InEKF] Steps: {len(dataset)}")
    print(f"[InEKF] RMSE (position): {rmse:.4f}")
    print(f"[InEKF] Runtime (filter only): {inekf_runtime:.3f} sec")

    out_dir = Path(args.output_dir)
    estimates_csv_path = out_dir / f"{dataset_name}_inekf_estimates.csv"
    saved_estimates_csv = save_estimates_to_csv(
        estimates_csv_path,
        estimates,
        pose_type,
        timestamps_ns=timestamps_ns,
        dt_values=dt_values,
    )
    print(f"[InEKF] Estimates CSV saved: {saved_estimates_csv}")

    vis_cfg = dict(inekf_cfg.get("visualization", {}))
    vis_cfg["estimator_label"] = "InEKF"

    plot_path = out_dir / f"{dataset_name}_inekf_trajectory.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path, visual_cfg=vis_cfg)
    print(f"[InEKF] Plot saved: {plot_path}")

    error_plot_path = out_dir / f"{dataset_name}_inekf_position_error_norm.png"
    plot_position_error_norm(estimates, gt, pose_type=pose_type, save_path=error_plot_path, visual_cfg=vis_cfg)
    print(f"[InEKF] Error plot saved: {error_plot_path}")

    anim_cfg = vis_cfg.get("animation", {})
    if vis_cfg.get("save_animation", False):
        anim_format = str(anim_cfg.get("format", "mp4")).lower()
        if anim_format not in {"mp4", "gif"}:
            raise ValueError("visualization.animation.format must be 'mp4' or 'gif'.")
        video_path = out_dir / f"{dataset_name}_inekf_trajectory.{anim_format}"
        saved = save_trajectory_animation(
            estimates,
            gt,
            pose_type=pose_type,
            save_path=video_path,
            fps=int(anim_cfg.get("fps", 20)),
            tail_length=int(anim_cfg.get("tail_length", 80)),
            moving_average_window=int(vis_cfg.get("error_moving_average_window", 50)),
            estimator_label="InEKF",
        )
        if saved:
            print(f"[InEKF] Animation saved: {video_path}")
        else:
            writer_name = "ffmpeg" if anim_format == "mp4" else "pillow"
            print(f"[InEKF] Animation skipped: {writer_name} writer is not available.")

    total_runtime = time.perf_counter() - total_start
    print(f"[InEKF] Runtime (total): {total_runtime:.3f} sec")


def _normalize_filter_config_for_pose(filter_cfg: dict, pose_type: str) -> None:
    measurement_cfg = filter_cfg.setdefault("measurement_model", {})
    evaluation_cfg = filter_cfg.setdefault("evaluation", {})
    visual_cfg = filter_cfg.setdefault("visualization", {})
    init_cfg = filter_cfg.setdefault("initialization", {})
    motion_cfg = filter_cfg.setdefault("motion_model", {})

    if pose_type != "3d":
        measurement_cfg.setdefault("position_indices", [0, 1])
        measurement_cfg["measurement_noise_diag"] = _resize_list(
            measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]),
            2,
        )
        evaluation_cfg.setdefault("position_indices", [0, 1])
        evaluation_cfg.setdefault("yaw_index", 2)
        visual_cfg.setdefault("position_indices", [0, 1])
        init_cfg["mean"] = _resize_list(init_cfg.get("mean", [0.0, 0.0, 0.0]), 3)
        init_cfg["cov_diag"] = _resize_list(init_cfg.get("cov_diag", [1.0, 1.0, 0.3]), 3)
        motion_cfg["process_noise_diag"] = _resize_list(
            motion_cfg.get("process_noise_diag", [0.02, 0.02, 0.005]),
            3,
        )
        return

    measurement_cfg["position_indices"] = [0, 1, 2]
    measurement_cfg["measurement_noise_diag"] = _resize_list(
        measurement_cfg.get("measurement_noise_diag", [0.7, 0.7, 0.7]),
        3,
    )
    evaluation_cfg["position_indices"] = [0, 1, 2]
    evaluation_cfg["yaw_index"] = 5
    visual_cfg["position_indices"] = [0, 1, 2]
    init_cfg["mean"] = _resize_list(init_cfg.get("mean", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6)
    init_cfg["cov_diag"] = _resize_list(
        init_cfg.get("cov_diag", [1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0]),
        9,
    )
    motion_cfg["process_noise_diag"] = _resize_list(
        motion_cfg.get("process_noise_diag", [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02]),
        9,
    )
    motion_cfg.setdefault("linear_input_type", "velocity")
    motion_cfg.setdefault("gravity", [0.0, 0.0, -9.81])


def _resize_list(values, dim: int) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == dim:
        return arr.tolist()
    if arr.size == 0:
        return np.zeros(dim, dtype=float).tolist()
    if arr.size == 1:
        return np.full(dim, float(arr.item()), dtype=float).tolist()
    out = np.zeros(dim, dtype=float)
    out[: min(dim, arr.size)] = arr[: min(dim, arr.size)]
    if arr.size < dim:
        out[arr.size :] = arr[-1]
    return out.tolist()



if __name__ == "__main__":
    main()
