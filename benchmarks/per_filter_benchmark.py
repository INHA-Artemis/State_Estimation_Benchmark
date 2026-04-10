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

PER_FILTER_BENCHMARK_CONFIG_PATH = Path(__file__).resolve().with_name("per_filter_benchmark.yaml")
DEFAULT_CONFIG_PATHS = {
    "ekf": Path("config/ekf.yaml"),
    "pf": Path("config/pf.yaml"),
    "ukf": Path("config/ukf.yaml"),
    "inekf": Path("config/inekf.yaml"),
}
FILTER_EXECUTION_ORDER = ("pf", "ukf", "ekf", "inekf")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/per_filter")
DEFAULT_DATASET_CONFIG_PATH = Path("config/dataset_config.yaml")


def main() -> None:
    cfg = load_yaml(PER_FILTER_BENCHMARK_CONFIG_PATH)
    filter_option = str(cfg.get("filter", "")).strip().lower()
    selected_filters = _resolve_filter_names(filter_option)

    dataset_config_path = _resolve_path(DEFAULT_DATASET_CONFIG_PATH)
    output_dir = _resolve_path(cfg.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_trials = int(cfg.get("num_trials", 10))
    dataset_seed_start = int(cfg.get("dataset_seed_start", 10))
    sync_filter_seed = bool(cfg.get("sync_filter_seed_with_dataset_seed", True))
    dataset_cases = list(cfg.get("dataset_cases", []))
    if not dataset_cases:
        raise ValueError("per_filter_benchmark.yaml must define at least one dataset_cases entry")

    filter_cases_map: dict[str, list[dict[str, Any]]] = {}
    total_runs = 0
    for filter_name in selected_filters:
        filter_cases = _load_filter_cases(cfg, filter_name)
        if not filter_cases:
            raise ValueError(
                "per_filter_benchmark.yaml must define at least one enabled filter case for "
                f"'{filter_name}' via '{filter_name}_cases' or 'filter_cases'"
            )
        filter_cases_map[filter_name] = filter_cases
        total_runs += len(dataset_cases) * len(filter_cases) * num_trials

    print(f"[PerFilterBenchmark] Filters         : {', '.join(selected_filters)}")
    print(f"[PerFilterBenchmark] Config          : {PER_FILTER_BENCHMARK_CONFIG_PATH}")
    print(f"[PerFilterBenchmark] Dataset cases   : {len(dataset_cases)}")
    print(
        "[PerFilterBenchmark] Filter cases    : "
        + ", ".join(f"{name}={len(filter_cases_map[name])}" for name in selected_filters)
    )
    print(f"[PerFilterBenchmark] Trials per pair : {num_trials}")
    print(f"[PerFilterBenchmark] Total runs      : {total_runs}")

    raw_rows_all: list[dict[str, Any]] = []

    with tqdm(total=total_runs, desc="Per-filter benchmark", unit="run") as pbar:
        for filter_name in selected_filters:
            filter_config_path = _resolve_path(DEFAULT_CONFIG_PATHS[filter_name])
            filter_cases = filter_cases_map[filter_name]
            print(f"[PerFilterBenchmark] Running filter  : {filter_name}")

            for dataset_case in dataset_cases:
                dataset_case_name = str(dataset_case.get("name", "dataset_case"))
                dataset_overrides = copy.deepcopy(dataset_case.get("dataset_overrides", {}))
                print(f"[PerFilterBenchmark] Dataset case    : {dataset_case_name}")

                for filter_case in filter_cases:
                    filter_case_name = str(filter_case.get("name", "filter_case"))
                    filter_overrides = copy.deepcopy(filter_case.get("filter_overrides", {}))

                    for trial in range(num_trials):
                        dataset_seed = dataset_seed_start + trial
                        dataset_cfg = load_yaml(dataset_config_path)
                        _deep_update(dataset_cfg, dataset_overrides)
                        dataset_cfg["seed"] = dataset_seed

                        pose_type, dataset_name, csv_path, dataset, gt, _dt, _timestamps_ns = prepare_dataset(dataset_cfg)
                        sequence_length = len(dataset)
                        mode = str(dataset_cfg.get("mode", "fused"))
                        dataset_type = str(dataset_cfg.get("dataset_type", "synthetic"))

                        filter_cfg = load_yaml(filter_config_path)
                        _deep_update(filter_cfg, filter_overrides)
                        _normalize_filter_config_for_pose(filter_name, filter_cfg, pose_type)
                        align_initialization_with_ground_truth(filter_cfg, gt, pose_type, mode)
                        if sync_filter_seed and filter_name == "pf":
                            filter_cfg["seed"] = dataset_seed

                        result = _run_filter(filter_name, dataset_cfg, filter_cfg, dataset, gt, pose_type)
                        raw_rows_all.append(
                            {
                                "filter": filter_name,
                                "dataset_case": dataset_case_name,
                                "filter_case": filter_case_name,
                                "trial": trial,
                                "dataset_seed": dataset_seed,
                                "dataset_type": dataset_type,
                                "dataset_name": dataset_name,
                                "dataset_csv": str(csv_path),
                                "pose_type": pose_type,
                                "mode": mode,
                                "sequence_length": sequence_length,
                                **result,
                            }
                        )
                        pbar.update(1)
                        pbar.set_postfix(filter=filter_name, dataset=dataset_case_name, case=filter_case_name, refresh=False)

    for filter_name in selected_filters:
        filter_raw_rows = [row for row in raw_rows_all if str(row.get("filter", "")).lower() == filter_name]
        summary_rows = _aggregate_rows(filter_raw_rows)

        raw_csv = output_dir / f"{filter_name}_per_filter_raw.csv"
        summary_csv = output_dir / f"{filter_name}_per_filter_summary.csv"
        _write_csv(raw_csv, filter_raw_rows)
        _write_csv(summary_csv, summary_rows)

        print(f"[PerFilterBenchmark] Raw results saved: {raw_csv}")
        print(f"[PerFilterBenchmark] Summary saved    : {summary_csv}")
        for row in summary_rows:
            label = f"{row['dataset_case']}::{row['filter_case']}"
            print(
                f"[PerFilterBenchmark] {label:<52} "
                f"mean_rmse={row['rmse_mean']:>7.4f} "
                f"std_rmse={row['rmse_std']:>7.4f} "
                f"mean_runtime={row['runtime_mean']:>6.3f} sec "
                f"n={int(row['num_trials']):>3d}"
            )


def _resolve_filter_names(filter_option: str) -> list[str]:
    if filter_option == "all":
        return list(FILTER_EXECUTION_ORDER)
    if filter_option in DEFAULT_CONFIG_PATHS:
        return [filter_option]
    raise ValueError("filter must be one of: ekf, pf, ukf, inekf, all")


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _run_filter(
    filter_name: str,
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
        "rmse_position": float(rmse),
        "runtime_sec": float(runtime),
    }


def _aggregate_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["filter"]), str(row["dataset_case"]), str(row["filter_case"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (filter_name, dataset_case, filter_case), rows in sorted(grouped.items()):
        rmse = np.array([float(row["rmse_position"]) for row in rows], dtype=float)
        runtime = np.array([float(row["runtime_sec"]) for row in rows], dtype=float)
        sample = rows[0]
        summary_rows.append(
            {
                "filter": filter_name,
                "dataset_case": dataset_case,
                "filter_case": filter_case,
                "dataset_type": sample["dataset_type"],
                "dataset_name": sample["dataset_name"],
                "pose_type": sample["pose_type"],
                "mode": sample["mode"],
                "sequence_length": sample["sequence_length"],
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


def _load_filter_cases(cfg: dict[str, Any], filter_name: str) -> list[dict[str, Any]]:
    specific_key = f"{filter_name}_cases"
    if specific_key in cfg:
        cases = list(cfg.get(specific_key, []))
    else:
        cases = list(cfg.get("filter_cases", []))
    return [case for case in cases if _case_is_enabled_for_filter(case, filter_name)]


def _case_is_enabled_for_filter(case: Any, filter_name: str) -> bool:
    if not isinstance(case, dict):
        return False
    if not bool(case.get("enabled", True)):
        return False

    supported_filters = case.get("filters", case.get("filter"))
    if supported_filters is None:
        return True
    if isinstance(supported_filters, str):
        return supported_filters.strip().lower() == filter_name
    if isinstance(supported_filters, (list, tuple, set)):
        normalized = {str(name).strip().lower() for name in supported_filters}
        return filter_name in normalized
    return False


def _normalize_filter_config_for_pose(filter_name: str, filter_cfg: dict[str, Any], pose_type: str) -> None:
    if filter_name in {"ekf", "ukf"}:
        normalize_position_filter_config_for_pose(filter_cfg, pose_type)
        return
    if filter_name == "pf":
        _normalize_pf_config_for_pose(filter_cfg, pose_type)
        return
    if filter_name == "inekf":
        _normalize_inekf_config_for_pose(filter_cfg, pose_type)
        return


def _normalize_pf_config_for_pose(filter_cfg: dict[str, Any], pose_type: str) -> None:
    measurement_cfg = filter_cfg.setdefault("measurement_model", {})

    target_dim = 3 if pose_type == "3d" else 2
    target_indices = [0, 1, 2] if target_dim == 3 else [0, 1]
    measurement_cfg["position_indices"] = target_indices

    measurement_noise = list(measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]))
    if len(measurement_noise) == 0:
        measurement_noise = [0.7] * target_dim
    elif len(measurement_noise) == 1:
        measurement_noise = measurement_noise * target_dim
    elif len(measurement_noise) < target_dim:
        measurement_noise.extend([measurement_noise[-1]] * (target_dim - len(measurement_noise)))
    elif len(measurement_noise) > target_dim:
        measurement_noise = measurement_noise[:target_dim]
    measurement_cfg["measurement_noise_diag"] = measurement_noise

    outlier_noise = list(measurement_cfg.get("outlier_noise_diag", [4.0, 4.0]))
    if len(outlier_noise) == 0:
        outlier_noise = [4.0] * target_dim
    elif len(outlier_noise) == 1:
        outlier_noise = outlier_noise * target_dim
    elif len(outlier_noise) < target_dim:
        outlier_noise.extend([outlier_noise[-1]] * (target_dim - len(outlier_noise)))
    elif len(outlier_noise) > target_dim:
        outlier_noise = outlier_noise[:target_dim]
    measurement_cfg["outlier_noise_diag"] = outlier_noise
def _normalize_inekf_config_for_pose(filter_cfg: dict[str, Any], pose_type: str) -> None:
    measurement_cfg = filter_cfg.setdefault("measurement_model", {})
    evaluation_cfg = filter_cfg.setdefault("evaluation", {})
    visual_cfg = filter_cfg.setdefault("visualization", {})
    init_cfg = filter_cfg.setdefault("initialization", {})
    motion_cfg = filter_cfg.setdefault("motion_model", {})

    if pose_type != "3d":
        measurement_cfg["position_indices"] = [0, 1]
        measurement_cfg["measurement_noise_diag"] = _resize_list(
            measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]),
            2,
        )
        evaluation_cfg["position_indices"] = [0, 1]
        evaluation_cfg["yaw_index"] = 2
        visual_cfg["position_indices"] = [0, 1]
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
