from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_REPO = PROJECT_ROOT / "compare_repos" / "invariant-ekf"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cpp_repo_exporters import export_invariant_ekf_input
from utils.benchmark_config import apply_benchmark_dataset_config, dataset_run_slug, default_benchmark_output_root
from utils.prepare_dataset import prepare_dataset
from utils.yaml_loader import load_yaml
from filters.invariant_kalman_filter import InvariantKalmanFilter
from filters.invariant_kalman_filter_15D import InvariantKalmanFilter15D
from utils.benchmark_visualization import write_metric_plots, write_trajectory_animation


@dataclass
class BenchmarkResult:
    family: str
    implementation: str
    algorithm: str
    status: str
    rmse_position: float | None
    error_variance: float | None
    runtime_sec: float | None
    steps: int
    notes: str = ""
    estimates: np.ndarray | None = None
    covariances: np.ndarray | None = None
    input_csv: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compare_repos/invariant-ekf on KAIST VIO.")
    parser.add_argument("--compare-config", default=str(PROJECT_ROOT / "config" / "compare.yaml"))
    parser.add_argument("--dataset-type", default=None, choices=["kaist_vio", "m2dgr"])
    parser.add_argument("--bag", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--imu-topic", default=None)
    parser.add_argument("--gt-topic", default=None)
    parser.add_argument("--gt-txt", default=None)
    parser.add_argument("--gnss-topic", default=None)
    parser.add_argument("--linear-source", default=None, choices=["gt_velocity", "accel"])
    measurement_group = parser.add_mutually_exclusive_group()
    measurement_group.add_argument("--use-pseudo-position-measurement", dest="use_pseudo_position_measurement", action="store_true", default=True)
    measurement_group.add_argument("--imu-only", dest="use_pseudo_position_measurement", action="store_false")
    parser.add_argument("--pseudo-position-stride", type=int, default=10)
    parser.add_argument("--pseudo-position-offset", type=int, default=0)
    parser.add_argument("--position-measurement-noise-std", nargs=3, type=float, default=[0.05, 0.05, 0.05])
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--animation-max-frames", type=int, default=600)
    parser.add_argument("--animation-fps", type=int, default=30)
    parser.add_argument(
        "--covariance-ellipsoid-sigma",
        type=float,
        default=5.0,
        help="Accepted for CLI compatibility; external InEKF runners do not currently output covariance ellipsoids.",
    )
    parser.add_argument("--runner", default="", help="Path to the invariant-ekf KAIST runner executable.")
    parser.add_argument(
        "--runner-command",
        default="{runner} --input {input} --output {output}",
        help="Command template. Available fields: {runner}, {input}, {output}.",
    )
    args = parser.parse_args()

    compare_cfg = load_yaml(Path(args.compare_config))
    apply_benchmark_dataset_config(args, compare_cfg, PROJECT_ROOT)
    if args.output_dir is None:
        args.output_dir = str(default_benchmark_output_root(PROJECT_ROOT, "invariant_ekf", args.dataset_type))
    output_dir = _resolve_run_output_dir(Path(args.output_dir), Path(args.bag), args.dataset_name, args.dataset_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_cfg = _effective_compare_config(compare_cfg, args)

    dataset_cfg = _build_dataset_config(args, output_dir)
    pose_type, dataset_name, dataset_csv, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)
    if pose_type != "3d":
        raise ValueError("invariant-ekf KAIST VIO benchmark expects 3D data.")
    if args.max_steps and args.max_steps > 0:
        dataset = dataset[: args.max_steps]
        gt = gt[: args.max_steps]
        timestamps_ns = timestamps_ns[: args.max_steps]
        if isinstance(dt, np.ndarray):
            dt = dt[: args.max_steps]
    _initialize_our_inekf_from_gt(compare_cfg, gt, dataset, args)

    filter_input_dir = output_dir / "filter_inputs"
    our_15d_dataset = _with_interpolated_position_measurements(dataset, gt, args.linear_source)
    our_9d_csv = _write_filter_input_csv(filter_input_dir / "our_inekf_9d.csv", dataset, gt, timestamps_ns)
    our_15d_csv = _write_filter_input_csv(filter_input_dir / "our_inekf_15d.csv", our_15d_dataset, gt, timestamps_ns)
    input_dir = export_invariant_ekf_input(output_dir / "repo_inputs" / "invariant_ekf", dataset, gt, timestamps_ns)
    estimates_csv = output_dir / "invariant_ekf_estimates.csv"
    results = [
        _run_our_inekf(compare_cfg, _estimator_dataset_config(dataset_cfg), dataset, gt, our_9d_csv),
        _run_our_inekf_15d(compare_cfg, _estimator_dataset_config(dataset_cfg), our_15d_dataset, gt, our_15d_csv),
        _run_invariant_ekf(args, input_dir, estimates_csv, gt),
    ]

    results_csv = output_dir / "invariant_ekf_kaist_vio_results.csv"
    _write_results_csv(results_csv, results)
    _write_estimates(output_dir / "estimates", results, timestamps_ns)
    metric_plot_paths = write_metric_plots(output_dir, results)
    animation_outputs = write_trajectory_animation(
        output_dir,
        results,
        gt,
        algorithm="inekf",
        max_frames=args.animation_max_frames,
        fps=args.animation_fps,
        covariance_sigma=args.covariance_ellipsoid_sigma,
    )
    _write_metadata(output_dir / "metadata.json", args, dataset_name, dataset_csv, input_dir, results, results_csv, metric_plot_paths, animation_outputs)

    print(f"[InvariantEkfBenchmark] Dataset CSV : {dataset_csv}")
    print(f"[InvariantEkfBenchmark] Our 9D CSV : {our_9d_csv}")
    print(f"[InvariantEkfBenchmark] Our 15D CSV: {our_15d_csv}")
    print(f"[InvariantEkfBenchmark] Repo input  : {input_dir}")
    print(f"[InvariantEkfBenchmark] Results     : {results_csv}")
    for algorithm, info in animation_outputs.items():
        animation_status = info["path"] if info["status"] == "ok" else info["status"]
        print(f"[InvariantEkfBenchmark] Animation {algorithm:<5}: {animation_status}")
    for result in results:
        if result.status == "ok":
            notes = f" ({result.notes})" if result.notes else ""
            print(
                f"[InvariantEkfBenchmark] {result.implementation:<14} inekf "
                f"rmse={result.rmse_position:.4f} var={result.error_variance:.6f} "
                f"runtime={result.runtime_sec:.3f}s{notes}"
            )
        else:
            print(f"[InvariantEkfBenchmark] {result.implementation:<14} inekf {result.status}: {result.notes}")


def _run_our_inekf(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    input_csv: Path,
) -> BenchmarkResult:
    return _run_body_frame_with_comparison(
        compare_cfg,
        "invariant_kalman_filter",
        ("ours", "Our InEKF", "inekf"),
        lambda cfg: _run_our_inekf_once(cfg, dataset_cfg, dataset, gt, input_csv),
    )


def _run_our_inekf_once(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    input_csv: Path,
) -> BenchmarkResult:
    try:
        estimator = InvariantKalmanFilter.from_configs(dataset_cfg, compare_cfg)
        start = time.perf_counter()
        estimates = []
        covariances = []
        for sample in dataset:
            estimates.append(estimator.step(sample.get("control"), sample.get("measurement"), float(sample.get("dt", 1.0))))
            covariances.append(_position_covariance(estimator))
        runtime = time.perf_counter() - start
        estimate_array = np.vstack(estimates) if estimates else np.zeros((0, 6), dtype=float)
        covariance_array = _stack_covariances(covariances)
        rmse, variance = _error_metrics(estimate_array, gt)
        return BenchmarkResult("ours", "Our InEKF", "inekf", "ok", rmse, variance, runtime, len(gt), estimates=estimate_array, covariances=covariance_array, input_csv=str(input_csv))
    except Exception as exc:
        return BenchmarkResult("ours", "Our InEKF", "inekf", "skipped", None, None, None, len(gt), f"{type(exc).__name__}: {exc}", input_csv=str(input_csv))


def _run_our_inekf_15d(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    input_csv: Path,
) -> BenchmarkResult:
    return _run_body_frame_with_comparison(
        compare_cfg,
        "invariant_kalman_filter_15d",
        ("ours", "Our InEKF 15D", "inekf"),
        lambda cfg: _run_our_inekf_15d_once(cfg, dataset_cfg, dataset, gt, input_csv),
    )


def _run_our_inekf_15d_once(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    input_csv: Path,
) -> BenchmarkResult:
    try:
        estimator = InvariantKalmanFilter15D.from_configs(dataset_cfg, compare_cfg)
        start = time.perf_counter()
        estimates = []
        covariances = []
        for sample in dataset:
            estimates.append(estimator.step(sample.get("control"), sample.get("measurement"), float(sample.get("dt", 1.0))))
            covariances.append(_position_covariance(estimator))
        runtime = time.perf_counter() - start
        estimate_array = np.vstack(estimates) if estimates else np.zeros((0, 6), dtype=float)
        covariance_array = _stack_covariances(covariances)
        rmse, variance = _error_metrics(estimate_array, gt)
        return BenchmarkResult("ours", "Our InEKF 15D", "inekf", "ok", rmse, variance, runtime, len(gt), estimates=estimate_array, covariances=covariance_array, input_csv=str(input_csv))
    except Exception as exc:
        return BenchmarkResult("ours", "Our InEKF 15D", "inekf", "skipped", None, None, None, len(gt), f"{type(exc).__name__}: {exc}", input_csv=str(input_csv))


def _run_body_frame_with_comparison(
    compare_cfg: dict[str, Any],
    config_key: str,
    result_identity: tuple[str, str, str],
    run_candidate: Any,
) -> BenchmarkResult:
    candidates = []
    for frame in ("world", "body"):
        candidate_cfg = _compare_config_with_translation_frame(compare_cfg, config_key, frame)
        candidates.append((frame, run_candidate(candidate_cfg)))

    ok_candidates = [
        (frame, result)
        for frame, result in candidates
        if result.status == "ok" and result.rmse_position is not None and np.isfinite(result.rmse_position)
    ]
    selected_frame, selected_result = (
        min(ok_candidates, key=lambda item: float(item[1].rmse_position))
        if ok_candidates
        else (None, None)
    )
    if selected_result is None:
        family, implementation, algorithm = result_identity
        notes = _frame_comparison_notes("none", candidates)
        return BenchmarkResult(family, implementation, algorithm, "skipped", None, None, None, candidates[0][1].steps if candidates else 0, notes)

    selected_result.notes = _frame_comparison_notes(str(selected_frame), candidates)
    return selected_result


def _compare_config_with_translation_frame(compare_cfg: dict[str, Any], config_key: str, frame: str) -> dict[str, Any]:
    cfg = copy.deepcopy(compare_cfg)
    motion_cfg = cfg.setdefault(config_key, {}).setdefault("motion_model", {})
    motion_cfg["translation_input_frame"] = frame
    return cfg


def _frame_comparison_notes(benchmark_frame: str, candidates: list[tuple[str, BenchmarkResult]]) -> str:
    details = []
    for frame, result in candidates:
        if result.status == "ok" and result.rmse_position is not None:
            details.append(f"{frame} rmse={result.rmse_position:.6g}")
        else:
            details.append(f"{frame} {result.status}")
    return f"benchmark_frame={benchmark_frame}; comparison: " + ", ".join(details)


def _run_invariant_ekf(args: argparse.Namespace, input_dir: Path, estimates_csv: Path, gt: np.ndarray) -> BenchmarkResult:
    if not COMPARE_REPO.exists():
        return _skipped(len(gt), f"repo missing at {COMPARE_REPO}")

    runner = _resolve_runner(args.runner)
    if runner is None:
        return _skipped(len(gt), "no runner executable found; pass --runner after building an invariant-ekf KAIST runner")

    estimates_csv.parent.mkdir(parents=True, exist_ok=True)
    command = args.runner_command.format(
        runner=str(runner.resolve()),
        input=str(input_dir.resolve()),
        output=str(estimates_csv.resolve()),
    )
    start = time.perf_counter()
    completed = subprocess.run(
        shlex.split(command),
        cwd=COMPARE_REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    runtime = time.perf_counter() - start
    if completed.returncode != 0:
        return _failed(len(gt), runtime, f"runner exited {completed.returncode}: {_compact_process_output(completed.stdout, completed.stderr)}")
    if not estimates_csv.exists():
        return _failed(len(gt), runtime, f"runner completed but did not write estimates CSV: {estimates_csv}")

    try:
        estimates = _load_estimates_csv(estimates_csv)
    except Exception as exc:
        return _failed(len(gt), runtime, f"could not read estimates CSV ({type(exc).__name__}: {exc})")
    rmse, variance = _error_metrics(estimates, gt)
    return BenchmarkResult("external", "invariant-ekf", "inekf", "ok", rmse, variance, runtime, len(gt), estimates=estimates, input_csv=str(input_dir))


def _resolve_runner(configured: str) -> Path | None:
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            COMPARE_REPO / "build" / "kaist_vio_runner",
            COMPARE_REPO / "build" / "invariant_ekf_kaist_vio_runner",
            COMPARE_REPO / "inekf" / "build" / "kaist_vio_runner",
            COMPARE_REPO / "inekf" / "build" / "bin" / "kaist_vio_runner",
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _build_dataset_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    mode = "imu_only"
    if args.use_pseudo_position_measurement:
        mode = (
            "fused"
            if args.dataset_type == "m2dgr"
            else f"fused_sampled_{max(1, int(args.pseudo_position_stride))}_{max(0, int(args.pseudo_position_offset))}"
        )
    common_cfg = {
        "dataset_name": args.dataset_name,
        "pose_type": "3d",
        "mode": mode,
        "generated_csv_path": str(output_dir / f"{args.dataset_name}_dataset.csv"),
        "use_imu": True,
        "use_gnss": args.dataset_type == "m2dgr",
        "use_position_measurement": bool(args.use_pseudo_position_measurement),
        "position_measurement_noise_model": "gaussian",
        "position_measurement_noise_std": list(args.position_measurement_noise_std),
        "gnss_noise_model": "gaussian",
        "gnss_noise_std": list(args.position_measurement_noise_std),
        "imu_noise_std": [0.0, 0.0],
        "imu_bias_std": [0.0, 0.0],
    }
    if args.dataset_type == "m2dgr":
        return {
            **common_cfg,
            "dataset_type": "m2dgr",
            "m2dgr_bag_path": str(Path(args.bag)),
            "m2dgr_gt_txt_path": str(Path(args.gt_txt)),
            "m2dgr_imu_topic": args.imu_topic,
            "m2dgr_gnss_topic": args.gnss_topic,
            "m2dgr_linear_source": args.linear_source,
            "m2dgr_use_gt_as_gnss": False,
            "measurement_source": "real_gnss_ublox_fix",
        }
    return {
        **common_cfg,
        "dataset_type": "kaist_vio",
        "rosbag_path": str(Path(args.bag)),
        "rosbag_imu_topic": args.imu_topic,
        "rosbag_gt_topic": args.gt_topic,
        "rosbag_linear_source": args.linear_source,
        "rosbag_use_gt_as_position_measurement": bool(args.use_pseudo_position_measurement),
        "pseudo_position_stride": max(1, int(args.pseudo_position_stride)),
        "pseudo_position_offset": max(0, int(args.pseudo_position_offset)),
    }


def _estimator_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(dataset_cfg)
    if str(cfg.get("mode", "fused")).startswith("fused_sampled_"):
        cfg["mode"] = "fused"
    return cfg


def _effective_compare_config(compare_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(compare_cfg)
    measurement_variance = np.square(np.asarray(args.position_measurement_noise_std, dtype=float)).tolist()

    inekf_cfg = cfg.setdefault("invariant_kalman_filter", {})
    motion_cfg = inekf_cfg.setdefault("motion_model", {})
    if args.linear_source == "gt_velocity":
        motion_cfg["translation_input_type"] = "velocity"
    meas_cfg = inekf_cfg.setdefault("measurement_model", {})
    meas_cfg["measurement_noise_diag"] = measurement_variance
    meas_cfg["innovation_gate_m"] = 0.0
    meas_cfg["mahalanobis_gate"] = 0.0

    inekf_15d_cfg = copy.deepcopy(inekf_cfg)
    inekf_15d_cfg["filter"] = {"name": "comparable_inekf_15d"}
    inekf_15d_cfg["initialization"]["cov_diag"] = [
        0.001,
        0.001,
        0.001,
        2.0,
        2.0,
        2.0,
        5.0,
        5.0,
        5.0,
        0.1,
        0.1,
        0.1,
        0.5,
        0.5,
        0.5,
    ]
    inekf_15d_cfg["motion_model"]["process_noise_diag"] = [
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-6,
        1.0e-6,
        1.0e-6,
    ]
    inekf_15d_cfg["motion_model"]["gravity"] = [0.0, 0.0, -9.81]
    inekf_15d_cfg["motion_model"]["accel_bias"] = [0.0, 0.0, 0.0]
    inekf_15d_cfg["motion_model"]["gyro_bias"] = [0.0, 0.0, 0.0]
    inekf_15d_cfg["measurement_model"]["measurement_noise_diag"] = measurement_variance
    inekf_15d_cfg["measurement_model"]["innovation_gate_m"] = 0.0
    inekf_15d_cfg["measurement_model"]["mahalanobis_gate"] = 0.0
    cfg["invariant_kalman_filter_15d"] = inekf_15d_cfg
    return cfg


def _initialize_our_inekf_from_gt(compare_cfg: dict[str, Any], gt: np.ndarray, dataset: list[dict], args: argparse.Namespace) -> None:
    if len(gt) == 0:
        return
    initial_pose = np.asarray(gt[0], dtype=float).reshape(-1)
    for key in ("invariant_kalman_filter", "invariant_kalman_filter_15d"):
        init_cfg = compare_cfg.setdefault(key, {}).setdefault("initialization", {})
        init_cfg["mean"] = initial_pose[:6].tolist()
        if args.linear_source == "gt_velocity":
            velocity = _initial_velocity_from_dataset(dataset)
            if velocity is not None:
                init_cfg["velocity_mean"] = velocity.tolist()


def _initial_velocity_from_dataset(dataset: list[dict]) -> np.ndarray | None:
    if not dataset:
        return None
    control = dataset[0].get("raw_control", dataset[0].get("control"))
    if control is None:
        return None
    control_vec = np.asarray(control, dtype=float).reshape(-1)
    if control_vec.size < 3:
        return None
    return control_vec[:3].copy()


def _write_filter_input_csv(path: Path, dataset: list[dict], gt: np.ndarray, timestamps_ns: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(gt, dtype=float)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    fieldnames = [
        "step",
        "timestamp_ns",
        "dt",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "has_position",
        "pos_x",
        "pos_y",
        "pos_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_roll",
        "gt_pitch",
        "gt_yaw",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(dataset):
            control = np.asarray(sample.get("raw_control", sample.get("control", np.zeros(6))), dtype=float).reshape(-1)
            if control.size < 6:
                control = np.pad(control, (0, 6 - control.size))
            measurement = sample.get("measurement")
            has_position = measurement is not None
            position = np.asarray(measurement if has_position else [np.nan, np.nan, np.nan], dtype=float).reshape(3)
            writer.writerow(
                {
                    "step": idx,
                    "timestamp_ns": int(timestamps[idx]),
                    "dt": float(sample.get("dt", 0.0)),
                    "ax": float(control[0]),
                    "ay": float(control[1]),
                    "az": float(control[2]),
                    "gx": float(control[3]),
                    "gy": float(control[4]),
                    "gz": float(control[5]),
                    "has_position": int(has_position),
                    "pos_x": float(position[0]),
                    "pos_y": float(position[1]),
                    "pos_z": float(position[2]),
                    "gt_x": float(gt[idx, 0]),
                    "gt_y": float(gt[idx, 1]),
                    "gt_z": float(gt[idx, 2]),
                    "gt_roll": float(gt[idx, 3]),
                    "gt_pitch": float(gt[idx, 4]),
                    "gt_yaw": float(gt[idx, 5]),
                }
            )
    return path


def _with_interpolated_position_measurements(dataset: list[dict], gt: np.ndarray, linear_source: str | None) -> list[dict]:
    measured_indices: list[int] = []
    measured_positions: list[np.ndarray] = []
    for idx, sample in enumerate(dataset):
        measurement = sample.get("measurement")
        if measurement is not None:
            measured_indices.append(idx)
            measured_positions.append(np.asarray(measurement, dtype=float).reshape(3))
    if not measured_positions:
        return dataset

    full_index = np.arange(len(dataset), dtype=float)
    measured_index = np.asarray(measured_indices, dtype=float)
    measured = np.vstack(measured_positions)
    interpolated = np.column_stack([np.interp(full_index, measured_index, measured[:, axis]) for axis in range(3)])

    use_accel_bias = str(linear_source).strip().lower() == "accel"
    accel_bias = _initial_accel_bias(dataset) if use_accel_bias else np.zeros(3, dtype=float)
    out: list[dict] = []
    for idx, sample in enumerate(dataset):
        copied = dict(sample)
        copied["measurement"] = interpolated[idx]
        control = sample.get("raw_control", sample.get("control"))
        if control is not None:
            control_vec = np.asarray(control, dtype=float).reshape(-1).copy()
            if control_vec.size >= 3:
                control_vec[0:3] -= accel_bias
            copied["control"] = control_vec
            copied["raw_control"] = control_vec
        out.append(copied)
    return out


def _initial_accel_bias(dataset: list[dict], window: int = 200) -> np.ndarray:
    controls = []
    for sample in dataset[:window]:
        control = sample.get("raw_control", sample.get("control"))
        if control is not None:
            control_vec = np.asarray(control, dtype=float).reshape(-1)
            if control_vec.size >= 3:
                controls.append(control_vec[0:3])
    if not controls:
        return np.zeros(3, dtype=float)
    return np.median(np.vstack(controls), axis=0) - np.array([0.0, 0.0, 9.81], dtype=float)


def _load_estimates_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row.get("roll", 0.0) or 0.0),
                    float(row.get("pitch", 0.0) or 0.0),
                    float(row.get("yaw", 0.0) or 0.0),
                ]
            )
    if not rows:
        raise ValueError("empty estimates CSV")
    return np.asarray(rows, dtype=float)


def _error_metrics(estimates: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    n = min(len(estimates), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    distances = np.linalg.norm(estimates[:n, :3] - gt[:n, :3], axis=1)
    return float(np.sqrt(np.mean(distances**2))), float(np.var(distances))


def _position_covariance(estimator: Any) -> np.ndarray | None:
    covariance = getattr(estimator, "P", None)
    if covariance is None:
        return None
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] < 9 or covariance.shape[1] < 9:
        return None
    return _finite_symmetric_covariance(covariance[6:9, 6:9])


def _finite_symmetric_covariance(covariance: np.ndarray) -> np.ndarray | None:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return None
    return 0.5 * (covariance + covariance.T)


def _stack_covariances(covariances: list[np.ndarray | None]) -> np.ndarray | None:
    if not covariances or all(cov is None for cov in covariances):
        return None
    empty = np.full((3, 3), np.nan, dtype=float)
    return np.stack([empty if cov is None else cov for cov in covariances], axis=0)


def _write_results_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["family", "implementation", "algorithm", "status", "rmse_position", "error_variance", "runtime_sec", "steps", "input_csv", "notes"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "family": result.family,
                    "implementation": result.implementation,
                    "algorithm": result.algorithm,
                    "status": result.status,
                    "rmse_position": "" if result.rmse_position is None else result.rmse_position,
                    "error_variance": "" if result.error_variance is None else result.error_variance,
                    "runtime_sec": "" if result.runtime_sec is None else result.runtime_sec,
                    "steps": result.steps,
                    "input_csv": result.input_csv,
                    "notes": result.notes,
                }
            )


def _write_estimates(output_dir: Path, results: list[BenchmarkResult], timestamps_ns: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.status != "ok" or result.estimates is None:
            continue
        path = output_dir / f"{_slug(result.implementation)}_{result.algorithm}_estimates.csv"
        estimates = result.estimates
        n = min(len(estimates), len(timestamps_ns))
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "timestamp_ns", "x", "y", "z", "roll", "pitch", "yaw"])
            for idx in range(n):
                writer.writerow([idx, int(timestamps_ns[idx]), *estimates[idx, :6].tolist()])


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    dataset_name: str,
    dataset_csv: Path,
    input_dir: Path,
    results: list[BenchmarkResult],
    results_csv: Path,
    metric_plot_paths: list[str],
    animation_outputs: dict[str, dict[str, str]],
) -> None:
    data = {
        "dataset_name": dataset_name,
        "dataset_csv": str(dataset_csv),
        "bag": str(args.bag),
        "repo": str(COMPARE_REPO),
        "repo_input": str(input_dir),
        "results_csv": str(results_csv),
        "implementation": "invariant-ekf",
        "algorithm": "inekf",
        "runner": str(args.runner),
        "run_mode": "fused" if args.use_pseudo_position_measurement else "imu_only",
        "linear_source": args.linear_source,
        "use_pseudo_position_measurement": bool(args.use_pseudo_position_measurement),
        "pseudo_position_stride": max(1, int(args.pseudo_position_stride)),
        "pseudo_position_offset": max(0, int(args.pseudo_position_offset)),
        "position_measurement_noise_std": list(args.position_measurement_noise_std),
        "plots": metric_plot_paths,
        "animations": {
            algorithm: {
                "path": Path(info["path"]).name,
                "status": info["status"],
                "max_frames": args.animation_max_frames,
                "fps": args.animation_fps,
                "trajectory_dimension": "3d",
                "covariance_ellipsoids": any(result.covariances is not None for result in results),
                "covariance_ellipsoid_sigma": float(args.covariance_ellipsoid_sigma),
            }
            for algorithm, info in animation_outputs.items()
        },
        "results": [
            {
                "implementation": result.implementation,
                "algorithm": result.algorithm,
                "status": result.status,
                "rmse_position": result.rmse_position,
                "error_variance": result.error_variance,
                "runtime_sec": result.runtime_sec,
                "input_csv": result.input_csv,
                "notes": result.notes,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _skipped(steps: int, notes: str) -> BenchmarkResult:
    return BenchmarkResult("external", "invariant-ekf", "inekf", "skipped", None, None, None, steps, notes)


def _failed(steps: int, runtime: float, notes: str) -> BenchmarkResult:
    return BenchmarkResult("external", "invariant-ekf", "inekf", "failed", None, None, runtime, steps, notes)


def _resolve_run_output_dir(output_root: Path, bag_path: Path, dataset_name: str, dataset_type: str = "kaist_vio") -> Path:
    run_slug = dataset_run_slug(str(bag_path), dataset_name, dataset_type)
    if output_root.name == run_slug:
        return output_root
    return output_root / run_slug


def _compact_process_output(stdout: str, stderr: str, max_chars: int = 500) -> str:
    text = " ".join("\n".join(part.strip() for part in (stdout, stderr) if part.strip()).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
