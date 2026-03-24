from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# 필터가 생성한 pose 추정값 시퀀스를 timestamp/dt와 함께 CSV로 저장한다.
def save_estimates_to_csv(
    csv_path: Path,
    estimates: np.ndarray,
    pose_type: str,
    timestamps_ns: np.ndarray,
    dt_values: np.ndarray,
) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    estimates = np.asarray(estimates, dtype=float)
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    dt_values = np.asarray(dt_values, dtype=float).reshape(-1)

    if estimates.shape[0] != timestamps_ns.shape[0] or estimates.shape[0] != dt_values.shape[0]:
        raise ValueError("estimates, timestamps_ns, and dt_values must have the same number of rows.")

    if pose_type == "2d":
        fieldnames = ["step", "timestamp_ns", "timestamp_sec", "dt", "est_x", "est_y", "est_yaw"]
    else:
        fieldnames = [
            "step",
            "timestamp_ns",
            "timestamp_sec",
            "dt",
            "est_x",
            "est_y",
            "est_z",
            "est_roll",
            "est_pitch",
            "est_yaw",
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for step, (ts_ns, dt, est) in enumerate(zip(timestamps_ns, dt_values, estimates, strict=False)):
            row = {
                "step": step,
                "timestamp_ns": int(ts_ns),
                "timestamp_sec": float(ts_ns) * 1e-9,
                "dt": float(dt),
            }
            if pose_type == "2d":
                row.update(
                    {
                        "est_x": float(est[0]),
                        "est_y": float(est[1]),
                        "est_yaw": float(est[2]),
                    }
                )
            else:
                row.update(
                    {
                        "est_x": float(est[0]),
                        "est_y": float(est[1]),
                        "est_z": float(est[2]),
                        "est_roll": float(est[3]),
                        "est_pitch": float(est[4]),
                        "est_yaw": float(est[5]),
                    }
                )
            writer.writerow(row)

    return csv_path
