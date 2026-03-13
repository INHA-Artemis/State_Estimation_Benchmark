# [협업 주석]
# Goal: estimation error 시각화 함수를 제공한다.
# What it does: 시간축 오차 곡선 및 요약 통계 플롯 로직을 추후 구현할 예정이다.
"""Error plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_error_plot(
    timestamps: np.ndarray,
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str | Path,
    position_indices: Sequence[int] = (0, 1),
    yaw_index: int | None = None,
    title: str = "Estimation Error",
) -> Path:
    """Save time-series error plot (position norm + optional yaw error)."""
    import matplotlib.pyplot as plt

    ts = np.asarray(timestamps, dtype=float)
    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if ts.ndim != 1:
        raise ValueError("timestamps must be 1D.")
    if est.ndim != 2 or gt.ndim != 2 or est.shape != gt.shape:
        raise ValueError("estimates and ground_truth must be same-shape 2D arrays.")
    if ts.size != est.shape[0]:
        raise ValueError("timestamps length must match sample count.")

    err = est - gt
    ix, iy = (int(position_indices[0]), int(position_indices[1]))
    mask = (
        np.isfinite(ts)
        & np.isfinite(est[:, ix])
        & np.isfinite(est[:, iy])
        & np.isfinite(gt[:, ix])
        & np.isfinite(gt[:, iy])
    )
    if not np.any(mask):
        raise ValueError("No finite samples available for error plotting.")

    ts = ts[mask]
    err = err[mask]
    pos_norm = np.linalg.norm(err[:, [ix, iy]], axis=1)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if yaw_index is None:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(ts, pos_norm, color="tab:red", linewidth=1.5, label="Position Error Norm")
        ax1.set_ylabel("Position Error [m]")
        ax1.set_xlabel("Time [s]")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="best")
        ax1.set_title(title)
    else:
        yaw_err = err[:, int(yaw_index)]
        yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(ts, pos_norm, color="tab:red", linewidth=1.5, label="Position Error Norm")
        ax1.set_ylabel("Position Error [m]")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="best")
        ax1.set_title(title)

        ax2.plot(ts, yaw_err, color="tab:purple", linewidth=1.4, label="Yaw Error")
        ax2.set_ylabel("Yaw Error [rad]")
        ax2.set_xlabel("Time [s]")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
