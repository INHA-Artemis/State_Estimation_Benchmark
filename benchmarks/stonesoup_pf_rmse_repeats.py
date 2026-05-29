from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class _NoOpProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, _n=1):
        return None

    def set_postfix_str(self, _value: str) -> None:
        return None


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - keeps the script usable without tqdm.
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else _NoOpProgress()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import stonesoup_kaist_vio_benchmark as ss_bench
from filters.particle_filter import ParticleFilter
from utils.benchmark_config import apply_benchmark_dataset_config, dataset_run_slug
from utils.prepare_dataset import prepare_dataset
from utils.yaml_loader import load_yaml


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_type: str
    dataset_name: str
    bag: str
    imu_topic: str
    linear_source: str
    gt_topic: str | None = None
    gt_txt: str | None = None
    gnss_topic: str | None = None


DATASETS = {
    "square": DatasetSpec(
        key="square",
        dataset_type="kaist_vio",
        dataset_name="kaist_vio_square",
        bag="datasets/KAIST_VIO/square.bag",
        imu_topic="/mavros/imu/data",
        gt_topic="/pose_transformed",
        linear_source="accel",
    ),
    "infinite": DatasetSpec(
        key="infinite",
        dataset_type="kaist_vio",
        dataset_name="kaist_vio_infinite",
        bag="datasets/KAIST_VIO/infinite.bag",
        imu_topic="/mavros/imu/data",
        gt_topic="/pose_transformed",
        linear_source="accel",
    ),
    "circle": DatasetSpec(
        key="circle",
        dataset_type="kaist_vio",
        dataset_name="kaist_vio_circle",
        bag="datasets/KAIST_VIO/circle.bag",
        imu_topic="/mavros/imu/data",
        gt_topic="/pose_transformed",
        linear_source="accel",
    ),
    "rotation": DatasetSpec(
        key="rotation",
        dataset_type="kaist_vio",
        dataset_name="kaist_vio_rotation",
        bag="datasets/KAIST_VIO/rotation.bag",
        imu_topic="/mavros/imu/data",
        gt_topic="/pose_transformed",
        linear_source="accel",
    ),
    "m2dgr": DatasetSpec(
        key="m2dgr",
        dataset_type="m2dgr",
        dataset_name="m2dgr_street_02",
        bag="datasets/M2DGR/street_02.bag",
        gt_txt="datasets/M2DGR/street_02.txt",
        imu_topic="/handsfree/imu",
        gnss_topic="/ublox/fix",
        linear_source="gt_velocity",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only Ours PF and Stone Soup PF repeatedly and save RMSE values for error-bar plots."
    )
    parser.add_argument("--compare-config", default=str(PROJECT_ROOT / "config" / "compare.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "stonesoup_pf_rmse_repeats"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", *DATASETS.keys()],
        default=["all"],
        help="Datasets to run. Use 'all' for square infinite circle rotation m2dgr.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Optional positive limit for quick checks.")
    parser.add_argument(
        "--use-pseudo-position-measurement",
        action="store_true",
        help="Use GT-sampled pseudo-position updates. Default is IMU-only.",
    )
    parser.add_argument("--pseudo-position-stride", type=int, default=10)
    parser.add_argument("--pseudo-position-offset", type=int, default=0)
    parser.add_argument(
        "--covariance-ellipsoid-sigma",
        type=float,
        default=1.0,
        help="Accepted for CLI compatibility; unused because this RMSE-only script draws no covariance ellipsoids.",
    )
    parser.add_argument(
        "--position-measurement-noise-std",
        nargs=3,
        type=float,
        default=[0.05, 0.05, 0.05],
        metavar=("SX", "SY", "SZ"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_dataset_specs(args.datasets)
    raw_rows: list[dict[str, Any]] = []
    total_filter_runs = len(selected) * max(0, int(args.runs)) * 2
    with tqdm(total=total_filter_runs, desc="PF repeat filters", unit="filter") as progress:
        for spec in selected:
            raw_rows.extend(_run_dataset_repeats(spec, args, output_dir, progress))

    raw_csv = output_dir / "pf_rmse_raw.csv"
    summary_csv = output_dir / "pf_rmse_summary.csv"
    _write_csv(raw_csv, raw_rows, _raw_fieldnames())
    _write_csv(summary_csv, _summary_rows(raw_rows), _summary_fieldnames())
    print(f"[PFRepeats] Raw RMSE CSV    : {raw_csv}")
    print(f"[PFRepeats] Summary RMSE CSV: {summary_csv}")


def _selected_dataset_specs(names: list[str]) -> list[DatasetSpec]:
    if "all" in names:
        return [DATASETS[key] for key in ("square", "infinite", "circle", "rotation", "m2dgr")]
    seen: set[str] = set()
    specs: list[DatasetSpec] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        specs.append(DATASETS[name])
    return specs


def _run_dataset_repeats(
    spec: DatasetSpec,
    args: argparse.Namespace,
    output_dir: Path,
    progress: Any,
) -> list[dict[str, Any]]:
    dataset_output_dir = output_dir / spec.key
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    bench_args = _benchmark_args(spec, args, dataset_output_dir)

    compare_cfg = load_yaml(Path(args.compare_config))
    apply_benchmark_dataset_config(bench_args, compare_cfg, PROJECT_ROOT)
    compare_cfg = ss_bench._effective_compare_config(compare_cfg, bench_args)

    run_output_dir = ss_bench._resolve_run_output_dir(bench_args)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = ss_bench._build_dataset_config(bench_args, run_output_dir)
    _pose_type, _dataset_name, _csv_path, dataset, gt, dt, _timestamps_ns = prepare_dataset(dataset_cfg)
    if bench_args.max_steps and bench_args.max_steps > 0:
        dataset = dataset[: bench_args.max_steps]
        gt = gt[: bench_args.max_steps]
        if isinstance(dt, np.ndarray):
            dt = dt[: bench_args.max_steps]

    ss_bench._initialize_filters_from_gt(compare_cfg, gt, dataset, bench_args)
    ss_bench._calibrate_cv_imu_bias(compare_cfg, dataset, bench_args)
    estimator_dataset_cfg = ss_bench._estimator_dataset_config(dataset_cfg)
    stone_soup = _load_stonesoup_particle_dependencies()

    print(f"[PFRepeats] Dataset {spec.key}: steps={len(gt)}, runs={args.runs}")
    rows: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        seed = int(args.start_seed + (run_index - 1) * args.seed_step)
        run_cfg = copy.deepcopy(compare_cfg)
        run_cfg.setdefault("particle_filter", {})["seed"] = seed

        progress.set_postfix_str(f"{spec.key} {run_index}/{args.runs} seed={seed} ours")
        progress_start = float(getattr(progress, "n", 0.0))
        our_result = _run_ours_pf(run_cfg, estimator_dataset_cfg, dataset, gt, progress)
        _finish_progress_unit(progress, progress_start)
        progress.set_postfix_str(f"{spec.key} {run_index}/{args.runs} seed={seed} stonesoup")
        progress_start = float(getattr(progress, "n", 0.0))
        stone_result = _run_stonesoup_pf(run_cfg, dataset, gt, stone_soup, progress)
        _finish_progress_unit(progress, progress_start)

        rows.append(_result_row(spec, run_index, seed, our_result))
        rows.append(_result_row(spec, run_index, seed, stone_result))
    return rows


def _benchmark_args(spec: DatasetSpec, args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        compare_config=args.compare_config,
        dataset_type=spec.dataset_type,
        bag=str((PROJECT_ROOT / spec.bag).resolve()),
        output_dir=str(output_dir),
        dataset_name=spec.dataset_name,
        imu_topic=spec.imu_topic,
        gt_topic=spec.gt_topic,
        gt_txt=str((PROJECT_ROOT / spec.gt_txt).resolve()) if spec.gt_txt else None,
        gnss_topic=spec.gnss_topic,
        linear_source=spec.linear_source,
        use_pseudo_position_measurement=bool(args.use_pseudo_position_measurement),
        pseudo_position_stride=args.pseudo_position_stride,
        pseudo_position_offset=args.pseudo_position_offset,
        position_measurement_noise_std=list(args.position_measurement_noise_std),
        max_steps=args.max_steps,
    )


def _load_stonesoup_particle_dependencies() -> dict[str, Any]:
    repo_path = ss_bench.COMPARE_REPOS_ROOT / "Stone-Soup"
    if not repo_path.exists():
        raise FileNotFoundError(f"Stone Soup repo missing at {repo_path}")

    with ss_bench._temporary_sys_path(repo_path):
        from stonesoup.models.control.linear import LinearControlModel
        from stonesoup.models.measurement.linear import LinearGaussian
        from stonesoup.models.transition.linear import LinearGaussianTimeInvariantTransitionModel
        from stonesoup.predictor.particle import ParticlePredictor
        from stonesoup.resampler.particle import ESSResampler, SystematicResampler
        from stonesoup.types.array import StateVectors
        from stonesoup.types.detection import Detection
        from stonesoup.types.hypothesis import SingleHypothesis
        from stonesoup.types.state import ParticleState, State
        from stonesoup.updater.particle import ParticleUpdater

    return {
        "LinearControlModel": LinearControlModel,
        "LinearGaussian": LinearGaussian,
        "LinearGaussianTimeInvariantTransitionModel": LinearGaussianTimeInvariantTransitionModel,
        "Detection": Detection,
        "SingleHypothesis": SingleHypothesis,
        "ParticleState": ParticleState,
        "State": State,
        "StateVectors": StateVectors,
        "ParticlePredictor": ParticlePredictor,
        "ParticleUpdater": ParticleUpdater,
        "ESSResampler": ESSResampler,
        "SystematicResampler": SystematicResampler,
    }


def _run_ours_pf(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    progress: Any,
) -> ss_bench.BenchmarkResult:
    try:
        estimator = ParticleFilter.from_configs(dataset_cfg, compare_cfg)
        with _benchmark_step_progress(progress):
            return ss_bench._time_stepwise_estimator("ours", "Our PF", "pf", estimator, dataset, gt)
    except Exception as exc:
        return ss_bench._skipped("ours", "Our PF", "pf", len(gt), f"{type(exc).__name__}: {exc}")


def _run_stonesoup_pf(
    compare_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
    stone_soup: dict[str, Any],
    progress: Any,
) -> ss_bench.BenchmarkResult:
    try:
        with _benchmark_step_progress(progress):
            return ss_bench._run_stonesoup_particle(
                compare_cfg["particle_filter"],
                dataset,
                gt,
                stone_soup,
                stone_soup["ParticlePredictor"],
                stone_soup["ParticleUpdater"],
                stone_soup["ESSResampler"],
                stone_soup["SystematicResampler"],
            )
    except Exception as exc:
        return ss_bench._skipped("external", "Stone Soup", "pf", len(gt), f"{type(exc).__name__}: {exc}")


class _benchmark_step_progress:
    def __init__(self, progress: Any) -> None:
        self.progress = progress
        self.start_value = float(getattr(progress, "n", 0.0))

    def __enter__(self):
        self.original_tqdm = ss_bench.tqdm
        ss_bench.tqdm = self._progress_tqdm
        return self

    def __exit__(self, exc_type, exc, tb):
        ss_bench.tqdm = self.original_tqdm
        return False

    def _progress_tqdm(self, iterable=None, *args, **kwargs):
        if iterable is None:
            return _NoOpProgress()
        try:
            total = len(iterable)
        except TypeError:
            total = 0
        increment = 1.0 / float(total) if total else 0.0
        for item in iterable:
            yield item
            if increment:
                self.progress.update(increment)


def _finish_progress_unit(progress: Any, start_value: float) -> None:
    current = float(getattr(progress, "n", 0.0))
    target = start_value + 1.0
    if target > current:
        progress.update(target - current)


def _result_row(
    spec: DatasetSpec,
    run_index: int,
    seed: int,
    result: ss_bench.BenchmarkResult,
) -> dict[str, Any]:
    return {
        "dataset": spec.key,
        "dataset_family": "m2dgr" if spec.dataset_type == "m2dgr" else "kaist_vio",
        "run_slug": dataset_run_slug(spec.bag, spec.dataset_name, spec.dataset_type),
        "repeat": run_index,
        "seed": seed,
        "family": result.family,
        "implementation": "Ours PF" if result.family == "ours" else result.implementation,
        "algorithm": result.algorithm,
        "status": result.status,
        "rmse_position": "" if result.rmse_position is None else result.rmse_position,
        "error_variance": "" if result.error_variance is None else result.error_variance,
        "runtime_sec": "" if result.runtime_sec is None else result.runtime_sec,
        "steps": result.steps,
        "notes": result.notes,
    }


def _summary_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["dataset"]), str(row["dataset_family"]), str(row["implementation"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, dataset_family, implementation), group in sorted(grouped.items()):
        values = np.asarray([float(row["rmse_position"]) for row in group], dtype=float)
        finite = values[np.isfinite(values)]
        n = int(finite.size)
        std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "dataset_family": dataset_family,
                "implementation": implementation,
                "algorithm": "pf",
                "n": n,
                "rmse_mean": float(np.mean(finite)) if n else "",
                "rmse_std": std if n else "",
                "rmse_stderr": std / float(np.sqrt(n)) if n else "",
                "rmse_min": float(np.min(finite)) if n else "",
                "rmse_max": float(np.max(finite)) if n else "",
                **_metric_summary(group, "runtime_sec", "runtime"),
            }
        )
    return rows


def _metric_summary(rows: list[dict[str, Any]], source_field: str, output_prefix: str) -> dict[str, Any]:
    values = np.asarray([_float_or_nan(row.get(source_field)) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    n = int(finite.size)
    std = float(np.std(finite, ddof=1)) if n > 1 else 0.0
    return {
        f"{output_prefix}_mean": float(np.mean(finite)) if n else "",
        f"{output_prefix}_std": std if n else "",
        f"{output_prefix}_stderr": std / float(np.sqrt(n)) if n else "",
        f"{output_prefix}_min": float(np.min(finite)) if n else "",
        f"{output_prefix}_max": float(np.max(finite)) if n else "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _raw_fieldnames() -> list[str]:
    return [
        "dataset",
        "dataset_family",
        "run_slug",
        "repeat",
        "seed",
        "family",
        "implementation",
        "algorithm",
        "status",
        "rmse_position",
        "error_variance",
        "runtime_sec",
        "steps",
        "notes",
    ]


def _summary_fieldnames() -> list[str]:
    return [
        "dataset",
        "dataset_family",
        "implementation",
        "algorithm",
        "n",
        "rmse_mean",
        "rmse_std",
        "rmse_stderr",
        "rmse_min",
        "rmse_max",
        "runtime_mean",
        "runtime_std",
        "runtime_stderr",
        "runtime_min",
        "runtime_max",
    ]


def _fmt_rmse(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6g}"


def _float_or_nan(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


if __name__ == "__main__":
    main()
