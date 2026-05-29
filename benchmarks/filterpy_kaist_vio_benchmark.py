from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - keeps the benchmark usable without tqdm.
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class _NoOpProgress:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def update(self, _n=1):
                    return None

            return _NoOpProgress()
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_REPOS_ROOT = PROJECT_ROOT / "compare_repos"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRIC_Y_AXIS_LIMITS = {
    "rmse_position": 0.15,
    "error_variance": 0.01,
    "runtime_sec": 60.0,
}

from filters.extended_kalman_filter import ExtendedKalmanFilter
from filters.unscented_kalman_filter import UnscentedKalmanFilter
from utils.benchmark_config import apply_benchmark_dataset_config, dataset_run_slug, default_benchmark_output_root
from utils.prepare_dataset import prepare_dataset
from utils.yaml_loader import load_yaml


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local comparable EKF/UKF with FilterPy EKF/UKF on KAIST VIO."
    )
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
    measurement_group.add_argument(
        "--use-pseudo-position-measurement",
        dest="use_pseudo_position_measurement",
        action="store_true",
        default=True,
        help="Use GT-derived synthetic position measurements. This is the default because KAIST_VIO has no GNSS topic.",
    )
    measurement_group.add_argument(
        "--imu-only",
        dest="use_pseudo_position_measurement",
        action="store_false",
        help="Disable pseudo-position updates and run pure IMU dead reckoning.",
    )
    parser.add_argument(
        "--pseudo-position-stride",
        type=int,
        default=10,
        help="Use one GT-sampled pseudo-position update every N GT samples when pseudo measurement is enabled.",
    )
    parser.add_argument("--pseudo-position-offset", type=int, default=0)
    parser.add_argument(
        "--position-measurement-noise-std",
        nargs=3,
        type=float,
        default=[0.05, 0.05, 0.05],
        metavar=("SX", "SY", "SZ"),
        help="Synthetic position measurement noise std, only used with --use-pseudo-position-measurement.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Optional positive limit for quick checks.")
    parser.add_argument("--animation-max-frames", type=int, default=600)
    parser.add_argument("--animation-fps", type=int, default=30)
    parser.add_argument(
        "--covariance-ellipsoid-sigma",
        type=float,
        default=5.0,
        help="Sigma scale for 3D position covariance ellipsoids in trajectory MP4s.",
    )
    parser.add_argument(
        "--hide-covariance-ellipsoids",
        action="store_true",
        help="Disable 3D covariance ellipsoids in trajectory MP4s.",
    )
    parser.add_argument("--skip-filterpy", action="store_true")
    args = parser.parse_args()

    compare_cfg = load_yaml(Path(args.compare_config))
    apply_benchmark_dataset_config(args, compare_cfg, PROJECT_ROOT)
    if args.output_dir is None:
        args.output_dir = str(default_benchmark_output_root(PROJECT_ROOT, "filterpy", args.dataset_type))
    compare_cfg = _effective_compare_config(compare_cfg, args)
    output_dir = _resolve_run_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _build_dataset_config(args, output_dir)
    pose_type, dataset_name, csv_path, dataset, gt, dt, timestamps_ns = prepare_dataset(dataset_cfg)
    if args.max_steps and args.max_steps > 0:
        dataset = dataset[: args.max_steps]
        gt = gt[: args.max_steps]
        timestamps_ns = timestamps_ns[: args.max_steps]
        if isinstance(dt, np.ndarray):
            dt = dt[: args.max_steps]
    _initialize_filters_from_gt(compare_cfg, gt, dataset, args)
    _calibrate_cv_imu_bias(compare_cfg, dataset, args)

    if pose_type != "3d":
        raise ValueError("This benchmark expects 3D KAIST VIO data.")
    estimator_dataset_cfg = _estimator_dataset_config(dataset_cfg)

    hyperparameter_snapshot = _hyperparameter_snapshot(compare_cfg)
    (output_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparameter_snapshot, indent=2),
        encoding="utf-8",
    )

    results: list[BenchmarkResult] = []
    results.extend(_run_local_comparable(compare_cfg, estimator_dataset_cfg, dataset, gt))

    if not args.skip_filterpy:
        results.extend(_run_filterpy(compare_cfg, dataset, gt))
    else:
        results.append(_skipped("external", "FilterPy", "ekf/ukf", len(gt), "disabled by --skip-filterpy"))

    _write_results_csv(output_dir / "filterpy_kaist_vio_results.csv", results)
    _write_estimates(output_dir / "estimates", results, timestamps_ns)
    metric_plot_paths = ["rmse_position.png", "error_variance.png", "runtime_sec.png"]
    _plot_metric_bars(output_dir / "rmse_position.png", results, "rmse_position", "Position RMSE [m]")
    _plot_metric_bars(output_dir / "error_variance.png", results, "error_variance", "Position Error Variance [m^2]")
    _plot_metric_bars(output_dir / "runtime_sec.png", results, "runtime_sec", "Total Runtime [s]")
    animation_outputs = _plot_algorithm_trajectory_animations(
        output_dir,
        results,
        gt,
        algorithms=("ekf", "ukf"),
        max_frames=args.animation_max_frames,
        fps=args.animation_fps,
        covariance_sigma=args.covariance_ellipsoid_sigma,
        draw_covariances=not args.hide_covariance_ellipsoids,
    )
    _write_run_metadata(
        output_dir / "metadata.json",
        args,
        dataset_name,
        csv_path,
        len(gt),
        results,
        animation_outputs,
        metric_plot_paths,
    )

    print(f"[FilterPyBenchmark] Dataset CSV : {csv_path}")
    print(f"[FilterPyBenchmark] Steps       : {len(gt)}")
    measurement_updates = sum(1 for sample in dataset if sample.get("measurement") is not None)
    print(f"[FilterPyBenchmark] Updates     : {measurement_updates}/{len(dataset)} position measurements")
    if args.use_pseudo_position_measurement:
        print(
            "[FilterPyBenchmark] Measurement : GT-sampled pseudo-position "
            f"(stride={max(1, args.pseudo_position_stride)}, not real GNSS)"
        )
    else:
        print("[FilterPyBenchmark] Measurement : none; IMU-only rosbag benchmark")
        print("[FilterPyBenchmark] Warning     : no EKF/UKF measurement update is applied; RMSE is IMU dead-reckoning drift.")
    print(f"[FilterPyBenchmark] Output dir  : {output_dir}")
    for algorithm, info in animation_outputs.items():
        animation_status = info["path"] if info["status"] == "ok" else info["status"]
        print(f"[FilterPyBenchmark] Animation {algorithm:<5}: {animation_status}")
    for result in results:
        if result.status == "ok":
            print(
                f"[FilterPyBenchmark] {result.implementation:<14} {result.algorithm:<5} "
                f"rmse={result.rmse_position:.4f} var={result.error_variance:.6f} "
                f"runtime={result.runtime_sec:.3f}s"
            )
        else:
            print(
                f"[FilterPyBenchmark] {result.implementation:<14} {result.algorithm:<5} "
                f"{result.status}: {result.notes}"
            )


def _resolve_run_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    run_slug = _run_output_slug(args)
    if output_root.name == run_slug:
        return output_root
    return output_root / run_slug


def _run_output_slug(args: argparse.Namespace) -> str:
    return dataset_run_slug(args.bag, args.dataset_name, args.dataset_type)


def _build_dataset_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    common_cfg = {
        "dataset_name": args.dataset_name,
        "pose_type": "3d",
        "mode": (
            f"fused_sampled_{max(1, int(args.pseudo_position_stride))}_{max(0, int(args.pseudo_position_offset))}"
            if args.use_pseudo_position_measurement
            else "imu_only"
        ),
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
    if args.dataset_type == "m2dgr":
        return {
            **common_cfg,
            "dataset_type": "m2dgr",
            "m2dgr_bag_path": str(Path(args.bag)),
            "m2dgr_gt_txt_path": str(Path(args.gt_txt)),
            "m2dgr_imu_topic": args.imu_topic,
            "m2dgr_gnss_topic": args.gnss_topic,
            "m2dgr_linear_source": args.linear_source,
            "m2dgr_use_gt_as_gnss": bool(args.use_pseudo_position_measurement),
        }
    return {
        **common_cfg,
        "dataset_type": "kaist_vio",
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
    }


def _estimator_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(dataset_cfg)
    mode = str(cfg.get("mode", "fused"))
    if mode.startswith("fused_sampled_"):
        cfg["mode"] = "fused"
    return cfg


def _effective_compare_config(compare_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(compare_cfg)
    measurement_variance = np.square(np.asarray(args.position_measurement_noise_std, dtype=float)).tolist()
    for key in ("kalman_filter", "extended_kalman_filter", "unscented_kalman_filter"):
        meas_cfg = cfg.setdefault(key, {}).setdefault("measurement_model", {})
        meas_cfg["measurement_noise_diag"] = measurement_variance

    return cfg


def _calibrate_cv_imu_bias(compare_cfg: dict[str, Any], dataset: list[dict], args: argparse.Namespace) -> None:
    if args.linear_source != "accel" or not dataset:
        return
    window = min(200, len(dataset))
    controls = np.vstack([np.asarray(sample.get("raw_control", sample.get("control")), dtype=float) for sample in dataset[:window]])
    accel_mean = np.mean(controls[:, 0:3], axis=0)

    motion_cfg = compare_cfg.setdefault("extended_kalman_filter", {}).setdefault("motion_model", {})
    gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
    expected_static_accel = -gravity
    accel_bias = accel_mean - expected_static_accel

    for key in ("extended_kalman_filter", "unscented_kalman_filter"):
        cv_motion_cfg = compare_cfg.setdefault(key, {}).setdefault("motion_model", {})
        cv_motion_cfg["control_input_type"] = "acceleration"
        cv_motion_cfg.setdefault("gravity", gravity.tolist())
        cv_motion_cfg["accel_bias"] = accel_bias.tolist()


def _initialize_filters_from_gt(compare_cfg: dict[str, Any], gt: np.ndarray, dataset: list[dict], args: argparse.Namespace) -> None:
    if len(gt) == 0:
        return

    initial_pose = np.asarray(gt[0], dtype=float).reshape(-1)
    initial_position = initial_pose[:3].tolist()
    initial_velocity = _initial_velocity_from_dataset(dataset) if args.linear_source == "gt_velocity" else None

    for key in ("extended_kalman_filter", "unscented_kalman_filter"):
        init_cfg = compare_cfg.setdefault(key, {}).setdefault("initialization", {})
        mean = list(init_cfg.get("mean", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        while len(mean) < 6:
            mean.append(0.0)
        mean[0:3] = initial_position
        if initial_velocity is not None:
            mean[3:6] = initial_velocity.tolist()
        init_cfg["mean"] = mean[:6]


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


def _run_local_comparable(
    compare_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
) -> list[BenchmarkResult]:
    constructors: list[tuple[str, str, type]] = [
        ("ekf", "Our EKF", ExtendedKalmanFilter),
        ("ukf", "Our UKF", UnscentedKalmanFilter),
    ]
    results: list[BenchmarkResult] = []
    for algorithm, name, cls in tqdm(constructors, desc="Our comparable filters", unit="filter"):
        try:
            estimator = cls.from_configs(dataset_cfg, compare_cfg)
            results.append(_time_stepwise_estimator("ours", name, algorithm, estimator, dataset, gt))
        except Exception as exc:
            results.append(_skipped("ours", name, algorithm, len(gt), f"{type(exc).__name__}: {exc}"))
    return results


def _run_filterpy(compare_cfg: dict[str, Any], dataset: list[dict], gt: np.ndarray) -> list[BenchmarkResult]:
    repo_path = COMPARE_REPOS_ROOT / "filterpy"
    if not repo_path.exists():
        return _skipped_external_suite("FilterPy", len(gt), f"repo missing at {repo_path}")

    with _temporary_sys_path(repo_path):
        try:
            from filterpy.kalman import ExtendedKalmanFilter as FPExtendedKalmanFilter
            from filterpy.kalman import MerweScaledSigmaPoints
            from filterpy.kalman import UnscentedKalmanFilter as FPUnscentedKalmanFilter
        except Exception as exc:
            return _skipped_external_suite(
                "FilterPy",
                len(gt),
                f"native import failed ({type(exc).__name__}: {exc})",
                algorithms=("ekf", "ukf"),
            )

    results: list[BenchmarkResult] = []
    results.append(_run_filterpy_ekf(FPExtendedKalmanFilter, compare_cfg, dataset, gt))
    results.append(_run_filterpy_ukf(MerweScaledSigmaPoints, FPUnscentedKalmanFilter, compare_cfg, dataset, gt))
    return results


def _run_filterpy_ekf(FPExtendedKalmanFilter: Any, compare_cfg: dict[str, Any], dataset: list[dict], gt: np.ndarray) -> BenchmarkResult:
    cfg = compare_cfg["extended_kalman_filter"]
    x0, P0, Q, H, R = _cv_matrices(cfg)
    motion_cfg = cfg.get("motion_model", {})

    def run(data: list[dict]) -> np.ndarray:
        ekf = FPExtendedKalmanFilter(dim_x=6, dim_z=3)
        ekf.x = x0.reshape(6, 1)
        ekf.P = P0.copy()
        ekf.Q = Q.copy()
        ekf.R = R.copy()

        def h_jacobian(_x):
            return H

        def h_x(x):
            return H @ x

        estimates = []
        covariances = []
        for sample in tqdm(data, desc="FilterPy EKF steps", unit="step"):
            dt = float(sample.get("dt", 1.0))
            ekf.F = _cv_F(dt)
            ekf.predict()
            ekf.x = ekf.x + _cv_control_vector(motion_cfg, sample.get("control"), dt).reshape(6, 1)
            measurement = sample.get("measurement")
            if measurement is not None:
                ekf.update(np.asarray(measurement, dtype=float).reshape(3, 1), h_jacobian, h_x, R=R)
            estimates.append(_pose_from_cv(np.asarray(ekf.x).reshape(-1)))
            covariances.append(_finite_symmetric_covariance(np.asarray(ekf.P, dtype=float)[0:3, 0:3]))
        return np.vstack(estimates), _stack_covariances(covariances)

    return _time_callable("external", "FilterPy", "ekf", run, dataset, gt)


def _run_filterpy_ukf(
    MerweScaledSigmaPoints: Any,
    FPUnscentedKalmanFilter: Any,
    compare_cfg: dict[str, Any],
    dataset: list[dict],
    gt: np.ndarray,
) -> BenchmarkResult:
    cfg = compare_cfg["unscented_kalman_filter"]
    x0, P0, Q, _H, R = _cv_matrices(cfg)
    motion_cfg = cfg.get("motion_model", {})
    sigma_cfg = cfg.get("sigma_points", {})
    current_control: dict[str, Any] = {"control": None}

    def fx(x, dt):
        return _cv_F(float(dt)) @ x + _cv_control_vector(motion_cfg, current_control["control"], float(dt))

    def hx(x):
        return x[:3]

    def run(data: list[dict]) -> np.ndarray:
        points = MerweScaledSigmaPoints(
            n=6,
            alpha=float(sigma_cfg.get("alpha", 0.5)),
            beta=float(sigma_cfg.get("beta", 2.0)),
            kappa=float(sigma_cfg.get("kappa", -3.0)),
        )
        ukf = FPUnscentedKalmanFilter(dim_x=6, dim_z=3, dt=1.0, fx=fx, hx=hx, points=points)
        ukf.x = x0.copy()
        ukf.P = P0.copy()
        ukf.Q = Q.copy()
        ukf.R = R.copy()
        estimates = []
        covariances = []
        for sample in tqdm(data, desc="FilterPy UKF steps", unit="step"):
            current_control["control"] = sample.get("control")
            ukf.predict(dt=float(sample.get("dt", 1.0)))
            measurement = sample.get("measurement")
            if measurement is not None:
                ukf.update(np.asarray(measurement, dtype=float).reshape(3), R=R)
            estimates.append(_pose_from_cv(ukf.x))
            covariances.append(_finite_symmetric_covariance(np.asarray(ukf.P, dtype=float)[0:3, 0:3]))
        return np.vstack(estimates), _stack_covariances(covariances)

    return _time_callable("external", "FilterPy", "ukf", run, dataset, gt)


def _skipped_external_suite(
    implementation: str,
    steps: int,
    notes: str,
    algorithms: tuple[str, ...] = ("ekf", "ukf"),
) -> list[BenchmarkResult]:
    return [_skipped("external", implementation, algorithm, steps, notes) for algorithm in algorithms]


def _time_estimator(
    family: str,
    implementation: str,
    algorithm: str,
    run_method: Callable[[list[dict]], np.ndarray],
    dataset: list[dict],
    gt: np.ndarray,
) -> BenchmarkResult:
    return _time_callable(family, implementation, algorithm, run_method, dataset, gt)


def _time_stepwise_estimator(
    family: str,
    implementation: str,
    algorithm: str,
    estimator: Any,
    dataset: list[dict],
    gt: np.ndarray,
    notes: str = "",
) -> BenchmarkResult:
    start = time.perf_counter()
    estimates = []
    covariances = []
    desc = f"{implementation} {algorithm.upper()} steps"
    for sample in tqdm(dataset, desc=desc, unit="step"):
        estimates.append(
            estimator.step(
                sample.get("control"),
                sample.get("measurement"),
                float(sample.get("dt", 1.0)),
            )
        )
        covariances.append(_position_covariance(estimator, algorithm))
    runtime = time.perf_counter() - start
    if estimates:
        estimate_array = np.vstack(estimates)
    else:
        estimate_array = np.zeros((0, 6), dtype=float)
    covariance_array = _stack_covariances(covariances)
    rmse, variance = _error_metrics(estimate_array, gt)
    return BenchmarkResult(
        family=family,
        implementation=implementation,
        algorithm=algorithm,
        status="ok",
        rmse_position=rmse,
        error_variance=variance,
        runtime_sec=float(runtime),
        steps=len(gt),
        notes=notes,
        estimates=estimate_array,
        covariances=covariance_array,
    )


def _time_callable(
    family: str,
    implementation: str,
    algorithm: str,
    run: Callable[[list[dict]], np.ndarray],
    dataset: list[dict],
    gt: np.ndarray,
) -> BenchmarkResult:
    start = time.perf_counter()
    output = run(dataset)
    runtime = time.perf_counter() - start
    if isinstance(output, tuple):
        estimates = output[0]
        covariances = output[1] if len(output) > 1 else None
    else:
        estimates = output
        covariances = None
    rmse, variance = _error_metrics(estimates, gt)
    return BenchmarkResult(
        family=family,
        implementation=implementation,
        algorithm=algorithm,
        status="ok",
        rmse_position=rmse,
        error_variance=variance,
        runtime_sec=float(runtime),
        steps=len(gt),
        estimates=estimates,
        covariances=covariances,
    )


def _position_covariance(estimator: Any, algorithm: str) -> np.ndarray | None:
    particles = getattr(estimator, "particles", None)
    weights = getattr(estimator, "weights", None)
    if particles is not None and weights is not None:
        particles_arr = np.asarray(particles, dtype=float)
        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        if particles_arr.ndim == 2 and particles_arr.shape[1] >= 3 and weights_arr.size == particles_arr.shape[0]:
            weight_sum = float(np.sum(weights_arr))
            if np.isfinite(weight_sum) and weight_sum > 0.0:
                weights_arr = weights_arr / weight_sum
                positions = particles_arr[:, 0:3]
                mean = np.average(positions, axis=0, weights=weights_arr)
                centered = positions - mean[None, :]
                cov = centered.T @ (centered * weights_arr[:, None])
                return _finite_symmetric_covariance(cov)

    P = getattr(estimator, "P", None)
    if P is None:
        return None
    P_arr = np.asarray(P, dtype=float)
    if P_arr.ndim != 2:
        return None
    if P_arr.shape[0] >= 3 and P_arr.shape[1] >= 3:
        return _finite_symmetric_covariance(P_arr[0:3, 0:3])
    return None


def _finite_symmetric_covariance(cov: np.ndarray) -> np.ndarray | None:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        return None
    return cov


def _stack_covariances(covariances: list[np.ndarray | None]) -> np.ndarray | None:
    if not covariances or all(cov is None for cov in covariances):
        return None
    empty = np.full((3, 3), np.nan, dtype=float)
    return np.stack([empty if cov is None else cov for cov in covariances], axis=0)


def _error_metrics(estimates: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    n = min(len(estimates), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    errors = estimates[:n, :3] - gt[:n, :3]
    error_norm = np.linalg.norm(errors, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    variance = float(np.var(error_norm))
    return rmse, variance


def _cv_matrices(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    init_cfg = cfg.get("initialization", {})
    motion_cfg = cfg.get("motion_model", {})
    meas_cfg = cfg.get("measurement_model", {})
    x0 = np.asarray(init_cfg.get("mean", [0, 0, 0, 0, 0, 0]), dtype=float).reshape(-1)
    if x0.size != 6:
        x0 = np.resize(x0, 6).astype(float)
    P0 = np.diag(np.asarray(init_cfg.get("cov_diag", [1, 1, 1, 1, 1, 1]), dtype=float).reshape(6))
    Q = np.diag(np.asarray(motion_cfg.get("process_noise_diag", [1e-3] * 6), dtype=float).reshape(6))
    H = np.zeros((3, 6), dtype=float)
    indices = np.asarray(meas_cfg.get("position_indices", [0, 1, 2]), dtype=int)
    for row, idx in enumerate(indices):
        H[row, idx] = 1.0
    R = np.diag(np.asarray(meas_cfg.get("measurement_noise_diag", [1, 1, 1]), dtype=float).reshape(3))
    return x0, P0, Q, H, R


def _cv_F(dt: float) -> np.ndarray:
    F = np.eye(6, dtype=float)
    F[0:3, 3:6] = np.eye(3, dtype=float) * float(dt)
    return F


def _cv_control_vector(motion_cfg: dict[str, Any], control: Any, dt: float) -> np.ndarray:
    b = np.zeros(6, dtype=float)
    if str(motion_cfg.get("control_input_type", "none")).lower() != "acceleration" or control is None:
        return b
    u = np.asarray(control, dtype=float).reshape(-1)
    if u.size < 3:
        return b
    gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(-1)
    accel_bias = np.asarray(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
    if gravity.size != 3:
        gravity = np.resize(gravity, 3)
    if accel_bias.size != 3:
        accel_bias = np.resize(accel_bias, 3)
    accel = u[:3] - accel_bias + gravity
    dt = float(dt)
    b[:3] = 0.5 * accel * dt**2
    b[3:6] = accel * dt
    return b


def _pose_from_cv(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1], x[2], 0.0, 0.0, 0.0], dtype=float)


def _diagonal_gaussian_logpdf(innovation: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.asarray(variance, dtype=float).reshape(1, -1)
    return -0.5 * (
        np.sum((innovation**2) / variance, axis=1)
        + np.sum(np.log(2.0 * np.pi * variance))
    )


def _std_from_variance(variance: list[float]) -> list[float]:
    return np.sqrt(np.asarray(variance, dtype=float)).tolist()


def _skipped(family: str, implementation: str, algorithm: str, steps: int, notes: str) -> BenchmarkResult:
    return BenchmarkResult(
        family=family,
        implementation=implementation,
        algorithm=algorithm,
        status="skipped",
        rmse_position=None,
        error_variance=None,
        runtime_sec=None,
        steps=steps,
        notes=notes,
    )


class _temporary_sys_path:
    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self.inserted = False

    def __enter__(self):
        if self.path not in sys.path:
            sys.path.insert(0, self.path)
            self.inserted = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.inserted:
            try:
                sys.path.remove(self.path)
            except ValueError:
                pass


def _write_results_csv(path: Path, results: list[BenchmarkResult]) -> None:
    rows = [
        {
            "family": r.family,
            "implementation": r.implementation,
            "algorithm": r.algorithm,
            "status": r.status,
            "rmse_position": "" if r.rmse_position is None else r.rmse_position,
            "error_variance": "" if r.error_variance is None else r.error_variance,
            "runtime_sec": "" if r.runtime_sec is None else r.runtime_sec,
            "steps": r.steps,
            "notes": r.notes,
        }
        for r in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_estimates(output_dir: Path, results: list[BenchmarkResult], timestamps_ns: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in tqdm(results, desc="Writing estimates", unit="file"):
        if result.status != "ok" or result.estimates is None:
            continue
        path = output_dir / f"{_slug(result.implementation)}_{result.algorithm}_estimates.csv"
        estimates = result.estimates
        n = len(estimates)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "timestamp_ns", "x", "y", "z", "roll", "pitch", "yaw"])
            for idx in range(n):
                ts = int(timestamps_ns[idx]) if idx < len(timestamps_ns) else ""
                writer.writerow([idx, ts, *estimates[idx, :6].tolist()])


def _plot_metric_bars(path: Path, results: list[BenchmarkResult], field: str, ylabel: str) -> None:
    ok_results = [r for r in results if r.status == "ok" and getattr(r, field) is not None]
    if not ok_results:
        return
    ok_results = _ordered_metric_results(ok_results)
    labels = [f"{r.implementation}\n{r.algorithm}" for r in ok_results]
    values = [float(getattr(r, field)) for r in ok_results]
    colors = [_implementation_color(r.implementation) for r in ok_results]

    fig_width = max(10, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(ylabel)
    _set_metric_y_axis(ax, field)
    _add_implementation_legend(ax, ok_results)
    _annotate_bars(ax, bars, values)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _set_metric_y_axis(ax, field: str) -> None:
    top = METRIC_Y_AXIS_LIMITS.get(field)
    if top is not None:
        ax.set_ylim(0.0, top)


def _implementation_color(implementation: str) -> str:
    if implementation.startswith("Our "):
        return "#2ca02c"
    if implementation in {"FilterPy", "invariant-ekf"}:
        return "#1f77b4"
    if implementation in {"Stone Soup", "drift"}:
        return "#ff7f0e"
    return "#7f7f7f"


def _add_implementation_legend(ax, results: list[BenchmarkResult]) -> None:
    handles = []
    labels = []
    for label, color in (
        ("Our filters", "#2ca02c"),
        ("FilterPy / invariant-ekf", "#1f77b4"),
        ("Stone Soup / drift", "#ff7f0e"),
    ):
        if any(_implementation_color(r.implementation) == color for r in results):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(label)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=min(3, len(handles)),
            fontsize=8,
            frameon=False,
        )


def _annotate_bars(ax, bars, values: list[float]) -> None:
    if not values:
        return
    is_log_like = ax.get_yscale() in {"log", "symlog"}
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-9)
    offset = max(y_range * 0.015, 1e-9)
    capped_label_y = y_max - y_range * 0.055
    for bar, value in zip(bars, values):
        if is_log_like and value > 0:
            y = value * 1.08
        else:
            y = min(bar.get_height() + offset, capped_label_y)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            _format_metric_value(value),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _format_metric_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-3:
        return f"{value:.2e}"
    if abs_value < 1:
        return f"{value:.4f}"
    if abs_value < 100:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _ordered_metric_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    algorithm_order = {"ekf": 0, "ukf": 1}
    family_order = {"ours": 0, "external": 1}
    implementation_order = {
        "Our EKF": 0,
        "Our UKF": 0,
        "FilterPy": 2,
        "Stone Soup": 3,
        "invariant-ekf": 2,
        "drift": 3,
    }
    return sorted(
        results,
        key=lambda r: (
            algorithm_order.get(r.algorithm, 99),
            family_order.get(r.family, 99),
            implementation_order.get(r.implementation, 50),
            r.implementation,
        ),
    )


def _trajectory_style_by_implementation() -> dict[str, tuple[str, str, float, float, int]]:
    return {
        "Our EKF": ("#2ca02c", "-", 2.0, 0.62, 3),
        "Our UKF": ("#2ca02c", "-", 2.0, 0.62, 3),
        "FilterPy": ("#1f77b4", "--", 2.3, 0.98, 6),
        "Stone Soup": ("#ff7f0e", ":", 2.7, 0.98, 7),
        "invariant-ekf": ("#1f77b4", "--", 2.3, 0.98, 6),
        "drift": ("#ff7f0e", ":", 2.7, 0.98, 7),
    }


def _plot_algorithm_trajectory_animations(
    output_dir: Path,
    results: list[BenchmarkResult],
    gt: np.ndarray,
    algorithms: tuple[str, ...],
    max_frames: int = 600,
    fps: int = 30,
    covariance_sigma: float = 2.0,
    draw_covariances: bool = True,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for algorithm in tqdm(algorithms, desc="Saving algorithm MP4s", unit="video"):
        path = output_dir / f"trajectory_comparison_{algorithm}.mp4"
        status = _plot_trajectory_animation_3d(
            path,
            results,
            gt,
            algorithm=algorithm,
            max_frames=max_frames,
            fps=fps,
            covariance_sigma=covariance_sigma,
            draw_covariances=draw_covariances,
        )
        outputs[algorithm] = {"path": str(path), "status": status}
    return outputs


def _plot_trajectory_animation_3d(
    path: Path,
    results: list[BenchmarkResult],
    gt: np.ndarray,
    algorithm: str,
    max_frames: int = 600,
    fps: int = 30,
    covariance_sigma: float = 2.0,
    draw_covariances: bool = True,
) -> str:
    ok_results = [
        r
        for r in results
        if r.status == "ok" and r.estimates is not None and r.algorithm == algorithm
    ]
    ok_results = sorted(ok_results, key=lambda r: (r.family != "ours", r.implementation))
    if not ok_results or len(gt) == 0:
        return f"skipped: no successful {algorithm} estimates to animate"
    if not animation.writers.is_available("ffmpeg"):
        return "skipped: ffmpeg writer is not available for mp4 output"

    n_steps = min([len(gt), *[len(r.estimates) for r in ok_results if r.estimates is not None]])
    if n_steps == 0:
        return "skipped: empty estimate arrays"

    frame_count = max(1, min(int(max_frames), n_steps))
    frame_indices = np.linspace(0, n_steps - 1, frame_count, dtype=int)
    gt_xyz = gt[:n_steps, :3]

    all_xyz = [gt_xyz]
    for result in ok_results:
        all_xyz.append(result.estimates[:n_steps, :3])
    stacked_xyz = np.vstack(all_xyz)
    mins = np.min(stacked_xyz, axis=0)
    maxs = np.max(stacked_xyz, axis=0)
    span = max(float(np.max(maxs - mins)), 1.0)
    pad = 0.08 * span
    centers = 0.5 * (mins + maxs)
    limits = [(float(center - 0.5 * span - pad), float(center + 0.5 * span + pad)) for center in centers]

    errors_by_name = {
        f"{r.implementation} {r.algorithm}": np.linalg.norm(r.estimates[:n_steps, :3] - gt[:n_steps, :3], axis=1)
        for r in ok_results
        if r.estimates is not None
    }
    max_error = max([float(np.max(errors)) for errors in errors_by_name.values()] + [1.0])

    fig = plt.figure(figsize=(13, 5.8))
    ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
    ax_err = fig.add_subplot(1, 2, 2)
    ax_traj.set_title(f"{algorithm.upper()} 3D Trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_zlabel("z [m]")
    ax_traj.set_xlim(*limits[0])
    ax_traj.set_ylim(*limits[1])
    ax_traj.set_zlim(*limits[2])
    ax_traj.grid(alpha=0.25)

    ax_err.set_title("Position Error")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("error [m]")
    ax_err.set_xlim(0, n_steps - 1)
    ax_err.set_ylim(0, max_error * 1.08)
    ax_err.grid(alpha=0.25)

    gt_line, = ax_traj.plot([], [], [], color="black", linewidth=2.2, label="GT")
    gt_point, = ax_traj.plot([], [], [], marker="o", color="black", markersize=4)
    estimate_lines = []
    estimate_points = []
    error_lines = []
    covariance_sources = []
    covariance_min_radius = max(0.015 * span, 0.03)
    style_by_implementation = _trajectory_style_by_implementation()
    for idx, result in enumerate(ok_results):
        label = f"{result.implementation} {result.algorithm}"
        color, linestyle, linewidth, alpha, zorder = style_by_implementation.get(
            result.implementation,
            (plt.rcParams["axes.prop_cycle"].by_key()["color"][idx % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])], "--", 2.0, 0.9, 5),
        )
        marker_size = 3 if result.family == "ours" else 4
        line, = ax_traj.plot(
            [],
            [],
            [],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            color=color,
            label=label,
            zorder=zorder,
        )
        point, = ax_traj.plot(
            [],
            [],
            [],
            marker="o",
            color=color,
            alpha=alpha,
            markersize=marker_size,
            zorder=zorder,
        )
        err_line, = ax_err.plot(
            [],
            [],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            color=color,
            label=label,
            zorder=zorder,
        )
        estimate_lines.append((line, result.estimates[:n_steps, :3]))
        estimate_points.append((point, result.estimates[:n_steps, :3]))
        error_lines.append((err_line, errors_by_name[label]))
        if draw_covariances and result.covariances is not None:
            covariance_sources.append(
                (
                    result.estimates[:n_steps, :3],
                    _display_covariance_sequence(
                        result.covariances[: min(n_steps, len(result.covariances)), :3, :3],
                        smooth=result.algorithm == "pf",
                    ),
                    color,
                    max(0.10, min(0.22, alpha * 0.25)),
                    zorder,
                )
            )

    ax_traj.legend(loc="best", fontsize=8)
    ax_err.legend(loc="best", fontsize=8)
    fig.tight_layout()
    covariance_surfaces: list[Poly3DCollection] = []

    def update(frame_number: int):
        end = int(frame_indices[frame_number]) + 1
        x_values = np.arange(end)
        while covariance_surfaces:
            covariance_surfaces.pop().remove()
        gt_line.set_data(gt_xyz[:end, 0], gt_xyz[:end, 1])
        gt_line.set_3d_properties(gt_xyz[:end, 2])
        gt_point.set_data([gt_xyz[end - 1, 0]], [gt_xyz[end - 1, 1]])
        gt_point.set_3d_properties([gt_xyz[end - 1, 2]])
        artists = [gt_line, gt_point]
        for line, xyz in estimate_lines:
            line.set_data(xyz[:end, 0], xyz[:end, 1])
            line.set_3d_properties(xyz[:end, 2])
            artists.append(line)
        for point, xyz in estimate_points:
            point.set_data([xyz[end - 1, 0]], [xyz[end - 1, 1]])
            point.set_3d_properties([xyz[end - 1, 2]])
            artists.append(point)
        for err_line, errors in error_lines:
            err_line.set_data(x_values, errors[:end])
            artists.append(err_line)
        cov_idx = end - 1
        for xyz, covariances, color, ellipsoid_alpha, zorder in covariance_sources:
            if cov_idx >= len(covariances):
                continue
            mesh = _covariance_ellipsoid_mesh(
                xyz[cov_idx],
                covariances[cov_idx],
                sigma=max(0.0, float(covariance_sigma)),
                min_radius=covariance_min_radius,
            )
            if mesh is None:
                continue
            x_mesh, y_mesh, z_mesh = mesh
            surface = ax_traj.plot_surface(
                x_mesh,
                y_mesh,
                z_mesh,
                color=color,
                alpha=ellipsoid_alpha,
                linewidth=0.0,
                shade=False,
                zorder=max(1, zorder - 1),
            )
            covariance_surfaces.append(surface)
            artists.append(surface)
        return artists

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=max(1, int(fps)), bitrate=1800)
    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / max(1, int(fps)), blit=False)
    try:
        with tqdm(total=len(frame_indices), desc="Saving MP4", unit="frame") as progress:
            def _progress_callback(_current_frame: int, _total_frames: int) -> None:
                progress.update(1)

            anim.save(path, writer=writer, dpi=140, progress_callback=_progress_callback)
    finally:
        plt.close(fig)
    return "ok"


def _covariance_ellipsoid_mesh(
    center: np.ndarray,
    covariance: np.ndarray,
    sigma: float = 2.0,
    min_radius: float = 0.0,
    resolution: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    center = np.asarray(center, dtype=float).reshape(3)
    covariance = 0.5 * (np.asarray(covariance, dtype=float).reshape(3, 3) + np.asarray(covariance, dtype=float).reshape(3, 3).T)
    if sigma <= 0.0 or not np.all(np.isfinite(center)) or not np.all(np.isfinite(covariance)):
        return None
    eigvals, eigvecs = np.linalg.eigh(covariance)
    radii = sigma * np.sqrt(np.clip(eigvals, 0.0, None))
    if not np.all(np.isfinite(radii)) or float(np.max(radii)) <= 1e-12:
        return None
    largest_radius = float(np.max(radii))
    if min_radius > 0.0 and largest_radius < min_radius:
        radii = radii * (float(min_radius) / largest_radius)
    u = np.linspace(0.0, 2.0 * np.pi, max(8, int(resolution)))
    v = np.linspace(0.0, np.pi, max(6, int(resolution // 2)))
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    points = center + (sphere * radii[None, None, :]) @ eigvecs.T
    return points[..., 0], points[..., 1], points[..., 2]


def _display_covariance_sequence(covariances: np.ndarray, smooth: bool = False) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3) or not smooth:
        return covariances
    smoothed = np.empty_like(covariances)
    previous: np.ndarray | None = None
    alpha = 0.92
    for idx, cov in enumerate(covariances):
        cov = 0.5 * (cov + cov.T)
        if not np.all(np.isfinite(cov)):
            smoothed[idx] = cov
            continue
        if previous is None:
            previous = cov
        else:
            previous = alpha * previous + (1.0 - alpha) * cov
        smoothed[idx] = previous
    return smoothed


def _write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    dataset_name: str,
    csv_path: Path,
    steps: int,
    results: list[BenchmarkResult],
    animation_outputs: dict[str, dict[str, str]],
    metric_plot_paths: list[str],
) -> None:
    data = {
        "dataset_name": dataset_name,
        "dataset_csv": str(csv_path),
        "bag": str(args.bag),
        "output_root": str(Path(args.output_dir)),
        "output_subdir": path.parent.name,
        "steps": steps,
        "benchmark_model": "comparable",
        "run_mode": "fused" if args.use_pseudo_position_measurement else "imu_only",
        "linear_source": args.linear_source,
        "initial_pose_source": "gt_first_sample",
        "measurement_source": (
            "gt_sampled_pseudo_position_not_real_gnss"
            if args.use_pseudo_position_measurement
            else "none_imu_only"
        ),
        "use_pseudo_position_measurement": bool(args.use_pseudo_position_measurement),
        "pseudo_position_stride": max(1, int(args.pseudo_position_stride)),
        "pseudo_position_offset": max(0, int(args.pseudo_position_offset)),
        "position_measurement_noise_std": list(args.position_measurement_noise_std),
        "results_csv": "filterpy_kaist_vio_results.csv",
        "plots": metric_plot_paths,
        "animations": {
            algorithm: {
                "path": Path(info["path"]).name,
                "status": info["status"],
                "max_frames": args.animation_max_frames,
                "fps": args.animation_fps,
                "trajectory_dimension": "3d",
                "covariance_ellipsoids": not args.hide_covariance_ellipsoids,
                "covariance_ellipsoid_sigma": float(args.covariance_ellipsoid_sigma),
            }
            for algorithm, info in animation_outputs.items()
        },
        "skipped": [
            {
                "implementation": r.implementation,
                "algorithm": r.algorithm,
                "notes": r.notes,
            }
            for r in results
            if r.status != "ok"
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hyperparameter_snapshot(compare_cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "extended_kalman_filter",
        "unscented_kalman_filter",
    ]
    return {"comparable": {key: compare_cfg.get(key, {}) for key in keys}}


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
