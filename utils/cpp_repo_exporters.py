from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def export_cpp_repo_inputs(
    output_dir: Path,
    dataset: list[dict],
    gt: np.ndarray,
    timestamps_ns: np.ndarray,
) -> dict[str, str]:
    """Export repo-specific inputs for the C++ comparison repositories."""
    export_dir = output_dir / "cpp_inputs"
    invariant_dir = export_invariant_ekf_input(export_dir / "invariant_ekf", dataset, gt, timestamps_ns)
    drift_csv = export_drift_input(export_dir / "drift.csv", dataset, gt, timestamps_ns)
    manifest = {
        "invariant_ekf": str(invariant_dir),
        "drift": str(drift_csv),
        "notes": {
            "invariant_ekf": (
                "Zurich/AGZ-style CSV folder for the original invariant-ekf DataLoader. "
                "Pseudo-position rows are converted to synthetic LLA GPS around a fixed origin."
            ),
            "drift": (
                "Flat CSV for a future drift CSV-to-measurement runner. Quaternion is reconstructed "
                "from benchmark GT roll/pitch/yaw."
            ),
        },
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": str(export_dir / "manifest.json"),
        "invariant_ekf": str(invariant_dir),
        "drift": str(drift_csv),
    }


def export_invariant_ekf_input(
    output_dir: Path,
    dataset: list[dict],
    gt: np.ndarray,
    timestamps_ns: np.ndarray,
) -> Path:
    log_dir = output_dir / "Log Files"
    log_dir.mkdir(parents=True, exist_ok=True)

    gt = _as_2d(gt, 6)
    timestamps_us = _timestamps_us(timestamps_ns, len(dataset))

    with (log_dir / "OnboardPose.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp_us",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "unused_0",
                "unused_1",
                "unused_2",
                "accel_bias_x",
                "accel_bias_y",
                "accel_bias_z",
                "source",
            ]
        )
        accel_bias = _estimate_initial_accel_bias(dataset)
        for idx, sample in enumerate(dataset):
            control = _control(sample)
            accel = control[0:3] - accel_bias
            writer.writerow(
                [
                    int(timestamps_us[idx]),
                    *_fmt(control[3:6]),
                    *_fmt(accel),
                    0.0,
                    0.0,
                    0.0,
                    *_fmt(accel_bias),
                    "kaist_vio",
                ]
            )

    with (log_dir / "OnboardGPS.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_us", "fix", "lat", "lon", "alt", "source"])
        positions = _interpolate_position_measurements(dataset, gt[:, 0:3], timestamps_us)
        repeat_count = 128
        for idx, position in enumerate(positions):
            for repeat_idx in range(repeat_count):
                lat, lon, alt = _local_enu_to_synthetic_lla(position + repeat_idx * 1.0e-9)
                writer.writerow([int(timestamps_us[idx]), 1, lat, lon, alt, "synthetic_lla_from_pseudo_position"])

    # Kept as an empty native-named file for folder readability. The KAIST
    # runner uses real-GPS mode, so DataLoader does not read this file.
    with (log_dir / "GroundTruthAGL.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "unused_0", "unused_1", "unused_2", "unused_3", "unused_4", "unused_5", "x", "y", "z", "source"])

    return output_dir


def export_drift_input(
    output_csv: Path,
    dataset: list[dict],
    gt: np.ndarray,
    timestamps_ns: np.ndarray,
    *,
    use_sample_velocity: bool = False,
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gt = _as_2d(gt, 6)
    timestamps_ns = _timestamps_ns_full(timestamps_ns, len(dataset))
    velocities = None if use_sample_velocity else _estimate_world_velocity(timestamps_ns, gt[:, 0:3])
    accel_bias = _estimate_initial_accel_bias(dataset)

    fieldnames = [
        "seq",
        "timestamp_ns",
        "timestamp_s",
        "dt",
        "qw",
        "qx",
        "qy",
        "qz",
        "gx",
        "gy",
        "gz",
        "ax",
        "ay",
        "az",
        "has_position",
        "pos_x",
        "pos_y",
        "pos_z",
        "has_velocity",
        "vel_x",
        "vel_y",
        "vel_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_roll",
        "gt_pitch",
        "gt_yaw",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(dataset):
            control = _control(sample)
            accel = control[0:3] - accel_bias
            quat = _rpy_to_quat_wxyz(gt[idx, 3:6])
            measurement = sample.get("measurement")
            has_position = measurement is not None
            position = np.asarray(measurement if has_position else [np.nan, np.nan, np.nan], dtype=float).reshape(3)
            velocity_measurement = sample.get("velocity_measurement")
            has_velocity = velocity_measurement is not None
            velocity = np.asarray(
                velocity_measurement if has_velocity else (velocities[idx] if velocities is not None else [np.nan, np.nan, np.nan]),
                dtype=float,
            ).reshape(3)
            has_velocity = bool(has_velocity or velocities is not None)
            writer.writerow(
                {
                    "seq": idx,
                    "timestamp_ns": int(timestamps_ns[idx]),
                    "timestamp_s": float(timestamps_ns[idx]) * 1e-9,
                    "dt": float(sample.get("dt", 0.0)),
                    "qw": float(quat[0]),
                    "qx": float(quat[1]),
                    "qy": float(quat[2]),
                    "qz": float(quat[3]),
                    "gx": float(control[3]),
                    "gy": float(control[4]),
                    "gz": float(control[5]),
                    "ax": float(accel[0]),
                    "ay": float(accel[1]),
                    "az": float(accel[2]),
                    "has_position": int(has_position),
                    "pos_x": float(position[0]),
                    "pos_y": float(position[1]),
                    "pos_z": float(position[2]),
                    "has_velocity": int(has_velocity and np.all(np.isfinite(velocity))),
                    "vel_x": float(velocity[0]),
                    "vel_y": float(velocity[1]),
                    "vel_z": float(velocity[2]),
                    "gt_x": float(gt[idx, 0]),
                    "gt_y": float(gt[idx, 1]),
                    "gt_z": float(gt[idx, 2]),
                    "gt_roll": float(gt[idx, 3]),
                    "gt_pitch": float(gt[idx, 4]),
                    "gt_yaw": float(gt[idx, 5]),
                }
            )
    return output_csv


def _control(sample: dict[str, Any]) -> np.ndarray:
    control = sample.get("raw_control", sample.get("control"))
    if control is None:
        return np.zeros(6, dtype=float)
    control_vec = np.asarray(control, dtype=float).reshape(-1)
    if control_vec.size < 6:
        raise ValueError("Expected 3D benchmark controls with [ax, ay, az, gx, gy, gz].")
    return control_vec[:6]


def _estimate_initial_accel_bias(dataset: list[dict], window: int = 200) -> np.ndarray:
    controls = []
    for sample in dataset[:window]:
        controls.append(_control(sample))
    if not controls:
        return np.zeros(3, dtype=float)
    accel_mean = np.median(np.vstack(controls)[:, 0:3], axis=0)
    return accel_mean - np.array([0.0, 0.0, 9.81], dtype=float)


def _as_2d(values: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < width:
        raise ValueError(f"Expected an array with at least {width} columns.")
    return array[:, :width]


def _timestamps_us(timestamps_ns: np.ndarray, length: int) -> np.ndarray:
    return np.rint(_timestamps_ns_full(timestamps_ns, length).astype(float) / 1000.0).astype(np.int64)


def _timestamps_ns_full(timestamps_ns: np.ndarray, length: int) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if timestamps.shape != (length,):
        raise ValueError("timestamps_ns must be a 1D array with the same length as dataset.")
    return timestamps


def _estimate_world_velocity(timestamps_ns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(positions) == 1:
        return np.zeros((1, 3), dtype=float)
    timestamps_s = np.asarray(timestamps_ns, dtype=float) * 1e-9
    raw_dt = np.diff(timestamps_s)
    positive_dt = raw_dt[raw_dt > 0.0]
    if positive_dt.size == 0:
        return np.zeros_like(positions, dtype=float)

    nominal_dt = float(np.median(positive_dt))
    min_dt = max(nominal_dt * 0.5, 1.0e-3)
    max_dt = max(nominal_dt * 3.0, min_dt)
    dt = np.clip(raw_dt, min_dt, max_dt)

    segment_velocity = np.diff(positions, axis=0) / dt[:, None]
    velocities = np.zeros_like(positions, dtype=float)
    velocities[0] = segment_velocity[0]
    velocities[-1] = segment_velocity[-1]
    velocities[1:-1] = 0.5 * (segment_velocity[:-1] + segment_velocity[1:])
    return velocities


def _interpolate_position_measurements(
    dataset: list[dict],
    fallback_positions: np.ndarray,
    timestamps_us: np.ndarray,
) -> np.ndarray:
    measured_indices: list[int] = []
    measured_positions: list[np.ndarray] = []
    for idx, sample in enumerate(dataset):
        measurement = sample.get("measurement")
        if measurement is not None:
            measured_indices.append(idx)
            measured_positions.append(np.asarray(measurement, dtype=float).reshape(3))

    if not measured_positions:
        return np.asarray(fallback_positions, dtype=float)[:, :3]

    measured_times = timestamps_us[np.asarray(measured_indices, dtype=int)].astype(float)
    measured = np.vstack(measured_positions)
    full_times = timestamps_us.astype(float)
    return np.column_stack(
        [
            np.interp(full_times, measured_times, measured[:, axis])
            for axis in range(3)
        ]
    )


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _local_enu_to_synthetic_lla(position: np.ndarray) -> tuple[float, float, float]:
    east, north, up = np.asarray(position, dtype=float).reshape(3)
    lat0_deg = 37.0
    lon0_deg = 127.0
    alt0_m = 100.0
    earth_radius_m = 6.371e6
    lat = lat0_deg + np.degrees(north / earth_radius_m)
    lon = lon0_deg + np.degrees(east / (earth_radius_m * np.cos(np.radians(lat0_deg))))
    alt = alt0_m + up
    return float(lat), float(lon), float(alt)


def _fmt(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]
