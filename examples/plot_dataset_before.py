from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.visualization import (
    control_trajectory_from_dataset,
    measurement_trajectory_from_dataset,
    plot_position_error_norm,
    plot_results,
)
from utils.yaml_loader import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT against before-filtering inputs.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--source", choices=("gnss", "imu", "both"), default="both")
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    pose_type, dataset_name, csv_path, dataset, gt, _dt, _timestamps_ns = prepare_dataset(dataset_cfg)
    out_dir = Path(args.output_dir)

    print(f"[Before] Pose type: {pose_type}")
    print(f"[Before] Dataset CSV: {csv_path}")
    print(f"[Before] Steps: {len(dataset)}")

    if args.source in ("gnss", "both"):
        _plot_before_source(
            label="GNSS",
            slug="gnss",
            before=measurement_trajectory_from_dataset(dataset, gt, pose_type),
            gt=gt,
            pose_type=pose_type,
            dataset_name=dataset_name,
            out_dir=out_dir,
        )

    if args.source in ("imu", "both"):
        _plot_before_source(
            label="IMU-only",
            slug="imu_only",
            before=control_trajectory_from_dataset(dataset, gt, pose_type),
            gt=gt,
            pose_type=pose_type,
            dataset_name=dataset_name,
            out_dir=out_dir,
        )


def _plot_before_source(
    label: str,
    slug: str,
    before,
    gt,
    pose_type: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    if before is None:
        print(f"[Before] {label} skipped: source is unavailable for this dataset/mode.")
        return

    rmse = compute_rmse(before, gt, pose_type=pose_type)
    vis_cfg = {"estimator_label": f"Before ({label})"}

    trajectory_path = out_dir / f"{dataset_name}_gt_before_{slug}_trajectory.png"
    plot_results(before, gt, pose_type=pose_type, save_path=trajectory_path, visual_cfg=vis_cfg)

    error_path = out_dir / f"{dataset_name}_gt_before_{slug}_position_error_norm.png"
    plot_position_error_norm(before, gt, pose_type=pose_type, save_path=error_path, visual_cfg=vis_cfg)

    print(f"[Before] {label} RMSE (position): {rmse:.4f}")
    print(f"[Before] {label} trajectory plot saved: {trajectory_path}")
    print(f"[Before] {label} error plot saved: {error_path}")


if __name__ == "__main__":
    main()
