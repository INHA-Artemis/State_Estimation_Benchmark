from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# examples/에서 실행해도 루트 모듈 import가 되도록 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.particle_filter import ParticleFilter
from utils.filter_initialization import align_initialization_with_ground_truth
from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.save_estimates import save_estimates_to_csv
from utils.visualization import plot_position_error_norm, plot_results, save_trajectory_animation
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run Particle Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--pf-config", default=str(PROJECT_ROOT / "config" / "pf.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    pf_cfg = load_yaml(Path(args.pf_config))

    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)

    if pose_type == "3d":
        measurement_cfg = pf_cfg.setdefault("measurement_model", {})
        position_indices = list(measurement_cfg.get("position_indices", [0, 1]))
        if len(position_indices) < 3:
            measurement_cfg["position_indices"] = [0, 1, 2]

        measurement_noise = list(measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]))
        if len(measurement_noise) == 1:
            measurement_noise = measurement_noise * 3
        elif len(measurement_noise) == 2:
            measurement_noise.append(measurement_noise[-1])
        measurement_cfg["measurement_noise_diag"] = measurement_noise

    align_initialization_with_ground_truth(pf_cfg, gt, pose_type, dataset_cfg.get("mode", "fused"))

    if np.isscalar(dt):
        dt_values = np.full(len(dataset), float(dt), dtype=float)
    else:
        dt_values = np.asarray(dt, dtype=float).reshape(-1)

    pf = ParticleFilter.from_configs(dataset_cfg, pf_cfg)

    pf_start = time.perf_counter()
    estimates = pf.run(dataset)
    pf_runtime = time.perf_counter() - pf_start

    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[PF] Pose type: {pose_type}")
    print(f"[PF] Dataset CSV: {csv_path}")
    print(f"[PF] Steps: {len(dataset)}")
    print(f"[PF] RMSE (position): {rmse:.4f}")
    print(f"[PF] Runtime (filter only): {pf_runtime:.3f} sec")

    out_dir = Path(args.output_dir)
    estimates_csv_path = out_dir / f"{dataset_name}_pf_estimates.csv"
    saved_estimates_csv = save_estimates_to_csv(
        estimates_csv_path,
        estimates,
        pose_type,
        timestamps_ns=timestamps_ns,
        dt_values=dt_values,
    )
    print(f"[PF] Estimates CSV saved: {saved_estimates_csv}")

    vis_cfg = pf_cfg.get("visualization", {})

    plot_path = out_dir / f"{dataset_name}_pf_trajectory.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path, visual_cfg=vis_cfg)
    print(f"[PF] Plot saved: {plot_path}")

    error_plot_path = out_dir / f"{dataset_name}_pf_position_error_norm.png"
    plot_position_error_norm(estimates, gt, pose_type=pose_type, save_path=error_plot_path, visual_cfg=vis_cfg)
    print(f"[PF] Error plot saved: {error_plot_path}")

    anim_cfg = vis_cfg.get("animation", {})
    if vis_cfg.get("save_animation", False):
        anim_format = str(anim_cfg.get("format", "mp4")).lower()
        if anim_format not in {"mp4", "gif"}:
            raise ValueError("visualization.animation.format must be 'mp4' or 'gif'.")
        video_path = out_dir / f"{dataset_name}_pf_trajectory.{anim_format}"
        saved = save_trajectory_animation(
            estimates,
            gt,
            pose_type=pose_type,
            save_path=video_path,
            fps=int(anim_cfg.get("fps", 20)),
            tail_length=int(anim_cfg.get("tail_length", 80)),
            moving_average_window=int(vis_cfg.get("error_moving_average_window", 50)),
        )
        if saved:
            print(f"[PF] Animation saved: {video_path}")
        else:
            writer_name = "ffmpeg" if anim_format == "mp4" else "pillow"
            print(f"[PF] Animation skipped: {writer_name} writer is not available.")

    total_runtime = time.perf_counter() - total_start
    print(f"[PF] Runtime (total): {total_runtime:.3f} sec")



if __name__ == "__main__":
    main()
