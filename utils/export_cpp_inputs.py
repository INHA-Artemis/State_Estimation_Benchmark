from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cpp_repo_exporters import export_cpp_repo_inputs
from utils.prepare_dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export KAIST VIO inputs for the invariant-ekf and drift C++ repositories."
    )
    parser.add_argument("--bag", default=str(PROJECT_ROOT / "datasets" / "KAIST_VIO" / "infinite.bag"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "cpp_inputs" / "kaist_vio"))
    parser.add_argument("--dataset-name", default="kaist_vio_infinite")
    parser.add_argument("--imu-topic", default="/mavros/imu/data")
    parser.add_argument("--gt-topic", default="/pose_transformed")
    parser.add_argument("--linear-source", default="accel", choices=["gt_velocity", "accel"])
    measurement_group = parser.add_mutually_exclusive_group()
    measurement_group.add_argument(
        "--use-pseudo-position-measurement",
        dest="use_pseudo_position_measurement",
        action="store_true",
        default=True,
        help="Use GT-derived synthetic position measurements.",
    )
    measurement_group.add_argument(
        "--imu-only",
        dest="use_pseudo_position_measurement",
        action="store_false",
        help="Disable pseudo-position rows.",
    )
    parser.add_argument("--pseudo-position-stride", type=int, default=10)
    parser.add_argument("--pseudo-position-offset", type=int, default=0)
    parser.add_argument(
        "--position-measurement-noise-std",
        nargs=3,
        type=float,
        default=[0.05, 0.05, 0.05],
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _build_dataset_config(args, output_dir)
    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)
    if pose_type != "3d":
        raise ValueError("C++ repo input export expects 3D KAIST VIO data.")

    if args.max_steps and args.max_steps > 0:
        dataset = dataset[: args.max_steps]
        gt = gt[: args.max_steps]
        timestamps_ns = timestamps_ns[: args.max_steps]
        if isinstance(dt, np.ndarray):
            dt = dt[: args.max_steps]

    cpp_input_paths = export_cpp_repo_inputs(output_dir, dataset, gt, timestamps_ns)
    metadata = {
        "dataset_name": dataset_name,
        "dataset_csv": str(csv_path),
        "bag": str(args.bag),
        "steps": len(gt),
        "run_mode": "fused" if args.use_pseudo_position_measurement else "imu_only",
        "linear_source": args.linear_source,
        "use_pseudo_position_measurement": bool(args.use_pseudo_position_measurement),
        "pseudo_position_stride": max(1, int(args.pseudo_position_stride)),
        "pseudo_position_offset": max(0, int(args.pseudo_position_offset)),
        "position_measurement_noise_std": list(args.position_measurement_noise_std),
        "cpp_inputs": cpp_input_paths,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[CppInputExport] Dataset CSV : {csv_path}")
    print(f"[CppInputExport] Manifest    : {cpp_input_paths['manifest']}")
    print(f"[CppInputExport] invariant-ekf: {cpp_input_paths['invariant_ekf']}")
    print(f"[CppInputExport] drift        : {cpp_input_paths['drift']}")
    print(f"[CppInputExport] Metadata     : {metadata_path}")
    print(f"[CppInputExport] Steps        : {len(gt)}")


def _build_dataset_config(args: argparse.Namespace, output_dir: Path) -> dict:
    return {
        "dataset_type": "kaist_vio",
        "dataset_name": args.dataset_name,
        "pose_type": "3d",
        "mode": (
            f"fused_sampled_{max(1, int(args.pseudo_position_stride))}_{max(0, int(args.pseudo_position_offset))}"
            if args.use_pseudo_position_measurement
            else "imu_only"
        ),
        "rosbag_path": str(Path(args.bag)),
        "rosbag_imu_topic": args.imu_topic,
        "rosbag_gt_topic": args.gt_topic,
        "rosbag_linear_source": args.linear_source,
        "rosbag_use_gt_as_position_measurement": bool(args.use_pseudo_position_measurement),
        "measurement_source": (
            "gt_sampled_pseudo_position_not_real_gnss"
            if args.use_pseudo_position_measurement
            else "none_imu_only"
        ),
        "pseudo_position_stride": max(1, int(args.pseudo_position_stride)),
        "pseudo_position_offset": max(0, int(args.pseudo_position_offset)),
        "generated_csv_path": str(output_dir / f"{args.dataset_name}_dataset.csv"),
        "use_imu": True,
        "use_gnss": False,
        "use_position_measurement": bool(args.use_pseudo_position_measurement),
        "position_measurement_noise_model": "gaussian",
        "position_measurement_noise_std": list(args.position_measurement_noise_std),
        "gnss_noise_model": "gaussian",
        "gnss_noise_std": list(args.position_measurement_noise_std),
        "imu_noise_std": [0.0, 0.0],
        "imu_bias_std": [0.0, 0.0],
    }


if __name__ == "__main__":
    main()
