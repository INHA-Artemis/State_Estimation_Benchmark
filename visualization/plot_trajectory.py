# [협업 주석]
# Goal: trajectory 결과 시각화 함수를 제공한다.
# What it does: estimate vs ground truth 궤적 비교 플로팅 로직을 추후 구현할 예정이다.
"""Trajectory plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_trajectory_plot(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str | Path,
    position_indices: Sequence[int] = (0, 1),
    gps_measurements: np.ndarray | None = None,
    title: str = "Trajectory",
) -> Path:
    """Save 2D trajectory comparison figure."""
    import matplotlib.pyplot as plt

    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if est.ndim != 2 or gt.ndim != 2 or est.shape != gt.shape:
        raise ValueError("estimates and ground_truth must be same-shape 2D arrays.")

    ix, iy = (int(position_indices[0]), int(position_indices[1]))
    mask = (
        np.isfinite(est[:, ix])
        & np.isfinite(est[:, iy])
        & np.isfinite(gt[:, ix])
        & np.isfinite(gt[:, iy])
    )
    if not np.any(mask):
        raise ValueError("No finite trajectory samples available for plotting.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt[mask, ix], gt[mask, iy], label="Ground Truth", color="black", linewidth=2.0)
    ax.plot(est[mask, ix], est[mask, iy], label="Estimate", color="tab:blue", linewidth=1.8)

    if gps_measurements is not None:
        gps = np.asarray(gps_measurements, dtype=float)
        if gps.ndim == 2 and gps.shape[1] >= 2:
            ax.scatter(gps[:, 0], gps[:, 1], s=8, alpha=0.35, label="GPS", color="tab:orange")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
