"""Entry point for running Particle Filter benchmark pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from datasets.factory import build_dataset_loader
from filters.pf import ParticleFilter
from models.measurement_model import PositionMeasurementModel
from models.motion_model import PlanarVelocityYawRateModel
from utils.config_loader import load_yaml_config

VALID_MODES = {"imu_only", "gps_only", "fused"}


def _default_state_dim(pose_type: str) -> int:
    return 6 if str(pose_type).lower() == "6d" else 3


def _build_motion_model(pf_cfg: dict[str, Any], state_dim: int) -> PlanarVelocityYawRateModel:
    motion_cfg = pf_cfg.get("motion_model", {})
    q_diag = motion_cfg.get("process_noise_diag", [0.05, 0.05, 0.01])
    q_diag_arr = np.asarray(q_diag, dtype=float)
    if q_diag_arr.size != state_dim:
        if q_diag_arr.size < state_dim:
            q_diag_arr = np.pad(q_diag_arr, (0, state_dim - q_diag_arr.size), constant_values=q_diag_arr[-1])
        else:
            q_diag_arr = q_diag_arr[:state_dim]

    return PlanarVelocityYawRateModel(
        state_dim=state_dim,
        process_noise_cov=np.diag(q_diag_arr),
        x_index=int(motion_cfg.get("x_index", 0)),
        y_index=int(motion_cfg.get("y_index", 1)),
        yaw_index=int(motion_cfg.get("yaw_index", 2 if state_dim == 3 else 5)),
    )


def _build_measurement_model(pf_cfg: dict[str, Any]) -> PositionMeasurementModel:
    meas_cfg = pf_cfg.get("measurement_model", {})
    indices = meas_cfg.get("position_indices", [0, 1])
    r_diag = np.asarray(meas_cfg.get("measurement_noise_diag", [1.0, 1.0]), dtype=float)
    return PositionMeasurementModel(position_indices=indices, measurement_noise_cov=np.diag(r_diag))


def run_pf(pf_config_path: Path, dataset_config_path: Path, mode_override: str | None = None) -> dict[str, np.ndarray]:
    pf_cfg = load_yaml_config(pf_config_path)
    ds_cfg = load_yaml_config(dataset_config_path)

    mode = str(mode_override or ds_cfg.get("mode", "fused")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got '{mode}'")

    dataset_loader = build_dataset_loader(ds_cfg)
    sequence = dataset_loader.load_sequence()

    state_dim = int(pf_cfg.get("state_dim", _default_state_dim(sequence.pose_type)))
    motion_model = _build_motion_model(pf_cfg, state_dim=state_dim)
    measurement_model = _build_measurement_model(pf_cfg) if mode in {"gps_only", "fused"} else None

    n_particles = int(pf_cfg.get("num_particles", 500))
    resample_ratio = float(pf_cfg.get("resample_threshold_ratio", 0.5))
    angle_indices = pf_cfg.get("angle_indices", [2 if state_dim == 3 else 5])
    pf = ParticleFilter(
        num_particles=n_particles,
        state_dim=state_dim,
        motion_model=motion_model,
        measurement_model=measurement_model,
        resample_threshold_ratio=resample_ratio,
        angle_indices=angle_indices,
        random_seed=pf_cfg.get("seed", None),
    )

    init_cfg = pf_cfg.get("initialization", {})
    init_mean = np.asarray(init_cfg.get("mean", [0.0] * state_dim), dtype=float)
    init_cov_diag = np.asarray(init_cfg.get("cov_diag", [1.0] * state_dim), dtype=float)
    pf.reset(initial_mean=init_mean, initial_cov=np.diag(init_cov_diag))

    est_states: list[np.ndarray] = []
    gt_states: list[np.ndarray] = []
    times: list[float] = []

    for step in sequence:
        if mode in {"imu_only", "fused"}:
            if step.imu is None:
                raise ValueError("IMU data required for selected mode but not present in dataset step.")
            pf.predict(step.imu, step.dt)
        elif mode == "gps_only":
            pf.predict(u=None, dt=step.dt)

        if mode in {"gps_only", "fused"} and step.gps is not None:
            pf.update(step.gps)

        est_states.append(pf.get_state())
        times.append(step.t)
        if step.gt_state is not None:
            gt_states.append(step.gt_state[:state_dim])

    estimates = np.asarray(est_states)
    timestamps = np.asarray(times)
    gt = np.asarray(gt_states) if gt_states else np.empty((0, state_dim))

    # Hook points for future integration:
    # - evaluation.compute_metrics(estimates, gt, ...)
    # - visualization.plot_trajectory(estimates, gt, ...)
    print(f"[PF] mode={mode}, steps={len(sequence)}, state_dim={state_dim}, particles={n_particles}")
    print(f"[PF] final_state={estimates[-1] if len(estimates) else 'N/A'}")

    return {"time": timestamps, "estimates": estimates, "ground_truth": gt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Particle Filter benchmark pipeline")
    parser.add_argument("--pf-config", type=Path, default=Path("config/pf.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("config/dataset_config.yaml"))
    parser.add_argument("--mode", type=str, default=None, choices=sorted(VALID_MODES))
    args = parser.parse_args()

    run_pf(
        pf_config_path=args.pf_config,
        dataset_config_path=args.dataset_config,
        mode_override=args.mode,
    )


if __name__ == "__main__":
    main()
