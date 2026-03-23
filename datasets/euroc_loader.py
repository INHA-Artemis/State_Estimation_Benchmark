from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from utils.rotation_utils import _rot_to_rpy


def load_euroc_dataset(dataset_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imu_csv = Path(dataset_cfg["euroc_imu_csv"])
    gt_csv = Path(dataset_cfg["euroc_gt_csv"])

    imu_timestamps, gyro = _load_imu_csv(imu_csv)
    gt_timestamps, gt, velocities = _load_ground_truth_csv(gt_csv)

    overlap_start = max(int(imu_timestamps[0]), int(gt_timestamps[0]))
    overlap_end = min(int(imu_timestamps[-1]), int(gt_timestamps[-1]))
    valid_mask = (gt_timestamps >= overlap_start) & (gt_timestamps <= overlap_end)
    if not np.any(valid_mask):
        raise ValueError("EuRoC IMU and ground-truth files do not share an overlapping time range.")

    gt_timestamps = gt_timestamps[valid_mask]
    gt = gt[valid_mask]
    velocities = velocities[valid_mask]

    imu_indices = np.searchsorted(imu_timestamps, gt_timestamps)
    imu_indices = np.clip(imu_indices, 1, len(imu_timestamps) - 1)
    prev_indices = imu_indices - 1
    use_prev = np.abs(gt_timestamps - imu_timestamps[prev_indices]) <= np.abs(gt_timestamps - imu_timestamps[imu_indices])
    imu_indices = np.where(use_prev, prev_indices, imu_indices)

    controls = np.zeros((len(gt_timestamps), 6), dtype=float)
    controls[:, :3] = velocities
    controls[:, 3:] = gyro[imu_indices]

    dt = np.zeros(len(gt_timestamps), dtype=float)
    if len(gt_timestamps) > 1:
        dt[1:] = np.diff(gt_timestamps) * 1e-9
        dt[0] = dt[1]
    else:
        dt[0] = float(dataset_cfg.get("dt", 0.005))

    measurements = _build_position_measurements(gt, dataset_cfg)
    return controls, measurements, gt, dt


def _load_imu_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    gyro_rows: list[list[float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamps.append(int(row["#timestamp [ns]"]))
            gyro_rows.append(
                [
                    float(row["w_RS_S_x [rad s^-1]"]),
                    float(row["w_RS_S_y [rad s^-1]"]),
                    float(row["w_RS_S_z [rad s^-1]"]),
                ]
            )

    if not timestamps:
        raise ValueError(f"No IMU rows found in {csv_path}.")
    return np.asarray(timestamps, dtype=np.int64), np.asarray(gyro_rows, dtype=float)


def _load_ground_truth_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    gt_rows: list[list[float]] = []
    velocity_rows: list[list[float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        for row in reader:
            timestamps.append(int(row["#timestamp"]))

            quat_w = float(row["q_RS_w []"])
            quat_x = float(row["q_RS_x []"])
            quat_y = float(row["q_RS_y []"])
            quat_z = float(row["q_RS_z []"])
            rpy = _quat_to_rpy(quat_w, quat_x, quat_y, quat_z)

            gt_rows.append(
                [
                    float(row["p_RS_R_x [m]"]),
                    float(row["p_RS_R_y [m]"]),
                    float(row["p_RS_R_z [m]"]),
                    float(rpy[0]),
                    float(rpy[1]),
                    float(rpy[2]),
                ]
            )
            velocity_rows.append(
                [
                    float(row["v_RS_R_x [m s^-1]"]),
                    float(row["v_RS_R_y [m s^-1]"]),
                    float(row["v_RS_R_z [m s^-1]"]),
                ]
            )

    if not timestamps:
        raise ValueError(f"No ground-truth rows found in {csv_path}.")
    return (
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(gt_rows, dtype=float),
        np.asarray(velocity_rows, dtype=float),
    )


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> np.ndarray:
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return _rot_to_rpy(R)


def _build_position_measurements(gt: np.ndarray, dataset_cfg: dict) -> np.ndarray:
    if not bool(dataset_cfg.get("euroc_use_gt_as_gnss", True)):
        return np.zeros((len(gt), 3), dtype=float)

    seed = int(dataset_cfg.get("seed", 10))
    rng = np.random.default_rng(seed)
    default_std = np.array([0.7, 0.7, 0.7], dtype=float)
    meas_std = np.asarray(dataset_cfg.get("gnss_noise_std", default_std), dtype=float).reshape(-1)
    if meas_std.size == 1:
        meas_std = np.full(3, float(meas_std.item()), dtype=float)
    elif meas_std.size == 2:
        meas_std = np.array([meas_std[0], meas_std[1], meas_std[1]], dtype=float)
    elif meas_std.size > 3:
        meas_std = meas_std[:3]

    return gt[:, :3] + rng.normal(0.0, meas_std, size=(len(gt), 3))
