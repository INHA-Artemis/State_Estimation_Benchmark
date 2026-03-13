# [협업 주석]
# Goal: 실험 단위 실행을 관리하는 runner를 제공한다.
# What it does: config 해석, dataset/filter wiring, 반복 실행 및 결과 저장 흐름을 추후 통합한다.
"""High-level benchmark runner for one estimator configuration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.estimator import EstimationResult, run_filter_sequence
from datasets.factory import build_dataset_loader
from evaluation.benchmark import build_run_summary, save_summary_json
from evaluation.metrics import compute_latency_only_metrics, compute_metrics
from filters.pf import ParticleFilter
from models.measurement_model import PositionMeasurementModel
from models.motion_model import PlanarVelocityYawRateModel
from visualization.plot_error import save_error_plot
from visualization.plot_trajectory import save_trajectory_plot
from visualization.realtime_plot import save_trajectory_animation

VALID_MODES = {"imu_only", "gps_only", "fused"}


def _default_state_dim(pose_type: str) -> int:
    return 6 if pose_type.lower() == "6d" else 3


def _fit_vector(values: list[float], state_dim: int, default_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == state_dim:
        return arr
    if arr.size == 0:
        return np.full((state_dim,), default_value, dtype=float)
    if arr.size < state_dim:
        return np.pad(arr, (0, state_dim - arr.size), constant_values=arr[-1])
    return arr[:state_dim]


def _build_motion_model(pf_cfg: dict[str, Any], state_dim: int) -> PlanarVelocityYawRateModel:
    motion_cfg = pf_cfg.get("motion_model", {})
    q_diag = _fit_vector(motion_cfg.get("process_noise_diag", [0.05, 0.05, 0.01]), state_dim, 0.01)
    yaw_default = 2 if state_dim == 3 else 5
    return PlanarVelocityYawRateModel(
        state_dim=state_dim,
        process_noise_cov=np.diag(q_diag),
        x_index=int(motion_cfg.get("x_index", 0)),
        y_index=int(motion_cfg.get("y_index", 1)),
        yaw_index=int(motion_cfg.get("yaw_index", yaw_default)),
    )


def _build_measurement_model(pf_cfg: dict[str, Any]) -> PositionMeasurementModel:
    meas_cfg = pf_cfg.get("measurement_model", {})
    indices = meas_cfg.get("position_indices", [0, 1])
    r_diag = np.asarray(meas_cfg.get("measurement_noise_diag", [1.0, 1.0]), dtype=float).reshape(-1)
    if r_diag.size != len(indices):
        raise ValueError("measurement_noise_diag size must match position_indices length.")
    return PositionMeasurementModel(position_indices=indices, measurement_noise_cov=np.diag(r_diag))


def _build_filter(
    pf_cfg: dict[str, Any],
    state_dim: int,
    motion_model: PlanarVelocityYawRateModel,
    measurement_model: PositionMeasurementModel | None,
) -> tuple[str, ParticleFilter]:
    filter_cfg = pf_cfg.get("filter", {})
    filter_name = str(filter_cfg.get("name", "pf")).lower()

    if filter_name != "pf":
        if filter_name in {"ekf", "inekf"}:
            raise NotImplementedError(
                f"filter='{filter_name}' is reserved for future integration. "
                "Runner already supports config-based switching, but only PF is implemented now."
            )
        raise ValueError(f"Unsupported filter name: {filter_name}")

    angle_default = [2] if state_dim == 3 else [5]
    pf = ParticleFilter(
        num_particles=int(pf_cfg.get("num_particles", 500)),
        state_dim=state_dim,
        motion_model=motion_model,
        measurement_model=measurement_model,
        resample_threshold_ratio=float(pf_cfg.get("resample_threshold_ratio", 0.5)),
        angle_indices=pf_cfg.get("angle_indices", angle_default),
        random_seed=pf_cfg.get("seed", None),
    )
    return filter_name, pf


def _prepare_output_dir(
    output_cfg: dict[str, Any],
    filter_name: str,
    mode: str,
    output_dir_override: str | Path | None = None,
) -> Path:
    base = Path(output_dir_override) if output_dir_override is not None else Path(output_cfg.get("dir", "outputs"))
    timestamped = bool(output_cfg.get("timestamped_subdir", True))
    if timestamped:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / f"{filter_name}_{mode}_{stamp}"
    else:
        path = base
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_estimation_benchmark(
    pf_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    mode_override: str | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Run one estimator benchmark and optionally save artifacts."""
    mode = str(mode_override or dataset_cfg.get("mode", "fused")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got '{mode}'")

    loader = build_dataset_loader(dataset_cfg)
    sequence = loader.load_sequence()

    state_dim = int(pf_cfg.get("state_dim", _default_state_dim(sequence.pose_type)))
    motion_model = _build_motion_model(pf_cfg, state_dim)
    measurement_model = _build_measurement_model(pf_cfg) if mode in {"gps_only", "fused"} else None
    filter_name, filter_obj = _build_filter(pf_cfg, state_dim, motion_model, measurement_model)

    init_cfg = pf_cfg.get("initialization", {})
    init_mean = _fit_vector(init_cfg.get("mean", []), state_dim, 0.0)
    init_cov_diag = _fit_vector(init_cfg.get("cov_diag", []), state_dim, 1.0)
    filter_obj.reset(initial_mean=init_mean, initial_cov=np.diag(init_cov_diag))

    result: EstimationResult = run_filter_sequence(filter_obj, sequence, mode, state_dim)

    eval_cfg = pf_cfg.get("evaluation", {})
    eval_pos_idx = tuple(int(i) for i in eval_cfg.get("position_indices", [0, 1]))
    yaw_default = 2 if state_dim == 3 else 5
    yaw_index = eval_cfg.get("yaw_index", yaw_default)

    valid_gt_mask = ~np.isnan(result.ground_truth).any(axis=1)
    if bool(np.any(valid_gt_mask)):
        metrics = compute_metrics(
            estimates=result.estimates[valid_gt_mask],
            ground_truth=result.ground_truth[valid_gt_mask],
            timestamps=result.timestamps[valid_gt_mask],
            position_indices=eval_pos_idx,
            yaw_index=int(yaw_index) if yaw_index is not None else None,
            step_latency_s=result.step_latency_s,
            predict_latency_s=result.predict_latency_s,
            update_latency_s=result.update_latency_s,
        )
    else:
        metrics = compute_latency_only_metrics(
            step_latency_s=result.step_latency_s,
            predict_latency_s=result.predict_latency_s,
            update_latency_s=result.update_latency_s,
        )
        metrics["warning"] = "Ground truth unavailable. Error metrics were skipped."

    output_cfg = pf_cfg.get("output", {})
    out_dir = _prepare_output_dir(output_cfg, filter_name, mode, output_dir_override=output_dir_override)
    artifacts: dict[str, str] = {}

    npz_path = out_dir / "run_data.npz"
    np.savez(
        npz_path,
        time=result.timestamps,
        estimates=result.estimates,
        ground_truth=result.ground_truth,
        gps=result.gps_measurements,
        step_latency_s=result.step_latency_s,
        predict_latency_s=result.predict_latency_s,
        update_latency_s=result.update_latency_s,
    )
    artifacts["run_data"] = str(npz_path)

    summary = build_run_summary(
        filter_name=filter_name,
        mode=mode,
        dataset_type=str(dataset_cfg.get("dataset_type", "unknown")),
        pose_type=sequence.pose_type,
        metrics=metrics,
    )
    summary_path = save_summary_json(summary, out_dir / "summary.json")
    artifacts["summary"] = str(summary_path)

    vis_cfg = pf_cfg.get("visualization", {})
    if bool(vis_cfg.get("enabled", True)):
        plot_pos_idx = tuple(int(i) for i in vis_cfg.get("position_indices", list(eval_pos_idx)))
        title_prefix = f"{filter_name.upper()} [{mode}]"
        try:
            traj_path = save_trajectory_plot(
                estimates=result.estimates,
                ground_truth=result.ground_truth,
                output_path=out_dir / "trajectory.png",
                position_indices=plot_pos_idx,
                gps_measurements=result.gps_measurements,
                title=f"{title_prefix} Trajectory",
            )
            artifacts["trajectory_plot"] = str(traj_path)
        except Exception as exc:
            artifacts["trajectory_plot_error"] = str(exc)

        try:
            err_path = save_error_plot(
                timestamps=result.timestamps,
                estimates=result.estimates,
                ground_truth=result.ground_truth,
                output_path=out_dir / "error.png",
                position_indices=plot_pos_idx,
                yaw_index=int(yaw_index) if yaw_index is not None else None,
                title=f"{title_prefix} Error",
            )
            artifacts["error_plot"] = str(err_path)
        except Exception as exc:
            artifacts["error_plot_error"] = str(exc)

        if bool(vis_cfg.get("save_animation", True)):
            anim_cfg = vis_cfg.get("animation", {})
            ext = str(anim_cfg.get("format", "gif")).lower().lstrip(".")
            anim_path = out_dir / f"trajectory_animation.{ext}"
            try:
                saved_anim = save_trajectory_animation(
                    timestamps=result.timestamps,
                    estimates=result.estimates,
                    ground_truth=result.ground_truth,
                    output_path=anim_path,
                    position_indices=plot_pos_idx,
                    gps_measurements=result.gps_measurements,
                    fps=int(anim_cfg.get("fps", 20)),
                    tail_length=int(anim_cfg.get("tail_length", 80)),
                    title=f"{title_prefix} Animation",
                )
                artifacts["animation"] = str(saved_anim)
            except Exception as exc:
                artifacts["animation_error"] = str(exc)

    return {
        "filter_name": filter_name,
        "mode": mode,
        "dataset_type": str(dataset_cfg.get("dataset_type", "unknown")),
        "pose_type": sequence.pose_type,
        "state_dim": state_dim,
        "result": result,
        "metrics": metrics,
        "output_dir": str(out_dir),
        "artifacts": artifacts,
    }
