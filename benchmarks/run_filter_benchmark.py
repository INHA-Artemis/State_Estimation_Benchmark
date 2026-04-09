from __future__ import annotations

import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.estimated_kalman_filter import ExtendedKalmanFilter
from filters.invariant_kalman_filter import InvariantKalmanFilter
from filters.particle_filter import ParticleFilter
from filters.unscented_kalman_filter import UnscentedKalmanFilter
from utils.filter_config import normalize_position_filter_config_for_pose
from utils.filter_initialization import align_initialization_with_ground_truth
from utils.math_utils import compute_rmse
from utils.prepare_dataset import prepare_dataset
from utils.yaml_loader import load_yaml

BENCHMARK_CONFIG_PATH = Path(__file__).resolve().with_name("benchmark_config.yaml")


def main() -> None:
    benchmark_cfg = load_yaml(BENCHMARK_CONFIG_PATH)

    dataset_config_path = _resolve_path("config/dataset_config.yaml")
    ekf_config_path = _resolve_path("config/ekf.yaml")
    pf_config_path = _resolve_path("config/pf.yaml")
    ukf_config_path = _resolve_path("config/ukf.yaml")
    inekf_config_path = _resolve_path("config/inekf.yaml")
    output_dir = _resolve_path("outputs/benchmarks")

    num_trials = int(benchmark_cfg.get("num_trials", 10))
    dataset_seed_start = int(benchmark_cfg.get("dataset_seed_start", 10))
    sync_pf_seed = bool(benchmark_cfg.get("sync_pf_seed_with_dataset_seed", True))

    pf_cases = list(benchmark_cfg.get("pf_cases", []))
    ukf_cases = list(benchmark_cfg.get("ukf_cases", []))
    inekf_cases = list(benchmark_cfg.get("inekf_cases", []))

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for trial in tqdm(range(num_trials), desc="Benchmark trials", unit="trial"):
        dataset_seed = dataset_seed_start + trial
        dataset_cfg = load_yaml(dataset_config_path)
        dataset_cfg["seed"] = dataset_seed

        pose_type, dataset_name, csv_path, dataset, gt, dt, _timestamps_ns = prepare_dataset(dataset_cfg)
        sequence_length = len(dataset)
        mode = str(dataset_cfg.get("mode", "fused"))
        dataset_type = str(dataset_cfg.get("dataset_type", "synthetic"))

        if trial == 0:
            print(
                "[Benchmark] Dataset setup    : "
                f"type={dataset_type} name={dataset_name} mode={mode} pose_type={pose_type} steps={sequence_length}"
            )
            print(f"[Benchmark] Dataset CSV      : {csv_path}")
            print(
                f"[Benchmark] Seed range       : start={dataset_seed_start} end={dataset_seed_start + num_trials - 1}"
            )

        ekf_cfg = load_yaml(ekf_config_path)
        normalize_position_filter_config_for_pose(ekf_cfg, pose_type)
        align_initialization_with_ground_truth(ekf_cfg, gt, pose_type, mode)
        ekf_result = _run_filter("ekf", "default", dataset_cfg, ekf_cfg, dataset, gt, pose_type)
        raw_rows.append(_build_raw_row(trial, dataset_seed, dataset_name, csv_path, sequence_length, mode, pose_type, ekf_result))

        for case in pf_cases:
            pf_cfg = load_yaml(pf_config_path)
            _deep_update(pf_cfg, case)
            _normalize_pf_config_for_pose(pf_cfg, pose_type)
            if sync_pf_seed:
                pf_cfg["seed"] = dataset_seed
            align_initialization_with_ground_truth(pf_cfg, gt, pose_type, mode)
            pf_result = _run_filter("pf", case["name"], dataset_cfg, pf_cfg, dataset, gt, pose_type)
            raw_rows.append(_build_raw_row(trial, dataset_seed, dataset_name, csv_path, sequence_length, mode, pose_type, pf_result))

        for case in ukf_cases:
            ukf_cfg = load_yaml(ukf_config_path)
            _deep_update(ukf_cfg, case)
            normalize_position_filter_config_for_pose(ukf_cfg, pose_type)
            align_initialization_with_ground_truth(ukf_cfg, gt, pose_type, mode)
            ukf_result = _run_filter("ukf", case["name"], dataset_cfg, ukf_cfg, dataset, gt, pose_type)
            raw_rows.append(_build_raw_row(trial, dataset_seed, dataset_name, csv_path, sequence_length, mode, pose_type, ukf_result))

        for case in inekf_cases:
            inekf_cfg = load_yaml(inekf_config_path)
            _deep_update(inekf_cfg, case)
            _normalize_inekf_config_for_pose(inekf_cfg, pose_type)
            align_initialization_with_ground_truth(inekf_cfg, gt, pose_type, mode)
            inekf_result = _run_filter("inekf", case["name"], dataset_cfg, inekf_cfg, dataset, gt, pose_type)
            raw_rows.append(_build_raw_row(trial, dataset_seed, dataset_name, csv_path, sequence_length, mode, pose_type, inekf_result))

    summary_rows.extend(_aggregate_rows(raw_rows))

    raw_csv = output_dir / "filter_benchmark_raw.csv"
    summary_csv = output_dir / "filter_benchmark_summary.csv"
    _write_csv(raw_csv, raw_rows)
    _write_csv(summary_csv, summary_rows)

    print(f"[Benchmark] Config used      : {BENCHMARK_CONFIG_PATH}")
    print(f"[Benchmark] Raw results saved: {raw_csv}")
    print(f"[Benchmark] Summary saved    : {summary_csv}")
    for row in summary_rows:
        label = f"{row['filter']}::{row['case']}"
        print(
            f"[Benchmark] {label:<38} "
            f"mean_rmse={row['rmse_mean']:>7.4f} "
            f"std_rmse={row['rmse_std']:>7.4f} "
            f"mean_runtime={row['runtime_mean']:>6.3f} sec "
            f"n={int(row['num_trials']):>3d}"
        )


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _run_filter(
    filter_name: str,
    case_name: str,
    dataset_cfg: dict,
    filter_cfg: dict,
    dataset: list[dict],
    gt: np.ndarray,
    pose_type: str,
) -> dict[str, Any]:
    filter_cfg = copy.deepcopy(filter_cfg)

    if filter_name == "ekf":
        estimator = ExtendedKalmanFilter.from_configs(dataset_cfg, filter_cfg)
    elif filter_name == "pf":
        estimator = ParticleFilter.from_configs(dataset_cfg, filter_cfg)
    elif filter_name == "ukf":
        estimator = UnscentedKalmanFilter.from_configs(dataset_cfg, filter_cfg)
    elif filter_name == "inekf":
        estimator = InvariantKalmanFilter.from_configs(dataset_cfg, filter_cfg)
    else:
        raise ValueError(f"Unsupported filter_name: {filter_name}")

    start = time.perf_counter()
    estimates = estimator.run(dataset)
    runtime = time.perf_counter() - start
    rmse = compute_rmse(estimates, gt, pose_type=pose_type)

    return {
        "filter": filter_name,
        "case": case_name,
        "rmse_position": float(rmse),
        "runtime_sec": float(runtime),
    }


def _build_raw_row(
    trial: int,
    dataset_seed: int,
    dataset_name: str,
    csv_path: Path,
    sequence_length: int,
    mode: str,
    pose_type: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "trial": trial,
        "dataset_seed": dataset_seed,
        "dataset_name": dataset_name,
        "dataset_csv": str(csv_path),
        "sequence_length": sequence_length,
        "mode": mode,
        "pose_type": pose_type,
    }
    row.update(result)
    return row


def _aggregate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["filter"]), str(row["case"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (filter_name, case_name), rows in sorted(grouped.items()):
        rmse = np.array([float(row["rmse_position"]) for row in rows], dtype=float)
        runtime = np.array([float(row["runtime_sec"]) for row in rows], dtype=float)
        summary_rows.append(
            {
                "filter": filter_name,
                "case": case_name,
                "num_trials": len(rows),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse)),
                "rmse_median": float(np.median(rmse)),
                "rmse_min": float(np.min(rmse)),
                "rmse_max": float(np.max(rmse)),
                "runtime_mean": float(np.mean(runtime)),
                "runtime_std": float(np.std(runtime)),
                "runtime_min": float(np.min(runtime)),
                "runtime_max": float(np.max(runtime)),
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _normalize_pf_config_for_pose(filter_cfg: dict[str, Any], pose_type: str) -> None:
    if pose_type != "3d":
        return

    measurement_cfg = filter_cfg.setdefault("measurement_model", {})
    position_indices = list(measurement_cfg.get("position_indices", [0, 1]))
    if len(position_indices) < 3:
        measurement_cfg["position_indices"] = [0, 1, 2]

    measurement_noise = list(measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]))
    if len(measurement_noise) == 1:
        measurement_noise = measurement_noise * 3
    elif len(measurement_noise) == 2:
        measurement_noise.append(measurement_noise[-1])
    elif len(measurement_noise) > 3:
        measurement_noise = measurement_noise[:3]
    measurement_cfg["measurement_noise_diag"] = measurement_noise

    outlier_noise = list(measurement_cfg.get("outlier_noise_diag", [4.0, 4.0]))
    if len(outlier_noise) == 1:
        outlier_noise = outlier_noise * 3
    elif len(outlier_noise) == 2:
        outlier_noise.append(outlier_noise[-1])
    elif len(outlier_noise) > 3:
        outlier_noise = outlier_noise[:3]
    measurement_cfg["outlier_noise_diag"] = outlier_noise


def _normalize_inekf_config_for_pose(filter_cfg: dict[str, Any], pose_type: str) -> None:
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


def _resize_list(values: Any, dim: int) -> list[float]:
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
