from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def save_dataset_to_csv(
    csv_path: Path,
    pose_type: str,
    dt: float | np.ndarray,
    controls: np.ndarray,
    measurements: np.ndarray,
    gt: np.ndarray,
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dt_values = np.asarray(dt, dtype=float)
    if dt_values.ndim == 0:
        dt_values = np.full(len(gt), float(dt_values), dtype=float)
    elif dt_values.shape != (len(gt),):
        raise ValueError("dt must be a scalar or a 1D array with the same length as gt.")

    if pose_type == "2d":
        fieldnames = [
            "step",
            "dt",
            "imu_speed",
            "imu_yaw_rate",
            "gnss_x",
            "gnss_y",
            "gt_x",
            "gt_y",
            "gt_yaw",
        ]
    else:
        fieldnames = [
            "step",
            "dt",
            "imu_dx",
            "imu_dy",
            "imu_dz",
            "imu_droll",
            "imu_dpitch",
            "imu_dyaw",
            "gnss_x",
            "gnss_y",
            "gnss_z",
            "gt_x",
            "gt_y",
            "gt_z",
            "gt_roll",
            "gt_pitch",
            "gt_yaw",
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(len(gt)):
            row = {"step": step, "dt": float(dt_values[step])}
            if pose_type == "2d":
                row.update(
                    {
                        "imu_speed": float(controls[step, 0]),
                        "imu_yaw_rate": float(controls[step, 1]),
                        "gnss_x": float(measurements[step, 0]),
                        "gnss_y": float(measurements[step, 1]),
                        "gt_x": float(gt[step, 0]),
                        "gt_y": float(gt[step, 1]),
                        "gt_yaw": float(gt[step, 2]),
                    }
                )
            else:
                row.update(
                    {
                        "imu_dx": float(controls[step, 0]),
                        "imu_dy": float(controls[step, 1]),
                        "imu_dz": float(controls[step, 2]),
                        "imu_droll": float(controls[step, 3]),
                        "imu_dpitch": float(controls[step, 4]),
                        "imu_dyaw": float(controls[step, 5]),
                        "gnss_x": float(measurements[step, 0]),
                        "gnss_y": float(measurements[step, 1]),
                        "gnss_z": float(measurements[step, 2]),
                        "gt_x": float(gt[step, 0]),
                        "gt_y": float(gt[step, 1]),
                        "gt_z": float(gt[step, 2]),
                        "gt_roll": float(gt[step, 3]),
                        "gt_pitch": float(gt[step, 4]),
                        "gt_yaw": float(gt[step, 5]),
                    }
                )
            writer.writerow(row)

    return csv_path


def load_dataset_from_csv(csv_path: Path, pose_type: str, mode: str) -> tuple[list[dict], np.ndarray]:
    dataset: list[dict] = []
    gt_rows: list[np.ndarray] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            dt = float(row["dt"])

            if pose_type == "2d":
                control = np.array([float(row["imu_speed"]), float(row["imu_yaw_rate"])], dtype=float)
                measurement = np.array([float(row["gnss_x"]), float(row["gnss_y"])], dtype=float)
                gt = np.array([float(row["gt_x"]), float(row["gt_y"]), float(row["gt_yaw"])], dtype=float)
            else:
                control = np.array(
                    [
                        float(row["imu_dx"]),
                        float(row["imu_dy"]),
                        float(row["imu_dz"]),
                        float(row["imu_droll"]),
                        float(row["imu_dpitch"]),
                        float(row["imu_dyaw"]),
                    ],
                    dtype=float,
                )
                measurement = np.array([float(row["gnss_x"]), float(row["gnss_y"]), float(row["gnss_z"])], dtype=float)
                gt = np.array(
                    [
                        float(row["gt_x"]),
                        float(row["gt_y"]),
                        float(row["gt_z"]),
                        float(row["gt_roll"]),
                        float(row["gt_pitch"]),
                        float(row["gt_yaw"]),
                    ],
                    dtype=float,
                )

            dataset.append(
                {
                    "control": None if mode == "gnss_only" else control,
                    "measurement": None if mode == "imu_only" else measurement,
                    "raw_control": control,
                    "raw_measurement": measurement,
                    "dt": dt,
                    "gt": gt,
                }
            )
            gt_rows.append(gt)

    if not gt_rows:
        gt = np.zeros((0, 3 if pose_type == "2d" else 6), dtype=float)
    else:
        gt = np.vstack(gt_rows)
    return dataset, gt
