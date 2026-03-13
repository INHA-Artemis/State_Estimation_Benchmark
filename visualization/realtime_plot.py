# [협업 주석]
# Goal: 실시간 상태 추정 결과 시각화 도구를 제공한다.
# What it does: time-step 업데이트 기반 trajectory/error plot 로직을 추후 구현할 예정이다.
"""Animation utilities for trajectory playback."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_trajectory_animation(
    timestamps: np.ndarray,
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str | Path,
    position_indices: Sequence[int] = (0, 1),
    gps_measurements: np.ndarray | None = None,
    fps: int = 20,
    tail_length: int = 60,
    title: str = "PF Trajectory Animation",
) -> Path:
    """
    Goal:
        trajectory 변화를 frame 단위 animation으로 저장한다.
    Input:
        timestamps, estimates, ground_truth는 animation 대상 배열이고, output_path는 저장 경로이다.
        position_indices, gps_measurements, fps, tail_length, title은 animation 표현 방식을 제어한다.
    Output:
        저장된 animation file의 Path를 반환한다. MP4 저장 실패 시 GIF fallback이 사용될 수 있다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ts = np.asarray(timestamps, dtype=float)
    est = np.asarray(estimates, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if ts.ndim != 1:
        raise ValueError("timestamps must be 1D.")
    if est.ndim != 2 or gt.ndim != 2 or est.shape != gt.shape:
        raise ValueError("estimates and ground_truth must be same-shape 2D arrays.")
    if ts.size != est.shape[0]:
        raise ValueError("timestamps length must match sample count.")

    ix, iy = (int(position_indices[0]), int(position_indices[1]))
    mask = (
        np.isfinite(ts)
        & np.isfinite(est[:, ix])
        & np.isfinite(est[:, iy])
        & np.isfinite(gt[:, ix])
        & np.isfinite(gt[:, iy])
    )
    if not np.any(mask):
        raise ValueError("No finite trajectory samples available for animation.")

    ts = ts[mask]
    est = est[mask]
    gt = gt[mask]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_min = float(min(np.min(est[:, ix]), np.min(gt[:, ix])))
    x_max = float(max(np.max(est[:, ix]), np.max(gt[:, ix])))
    y_min = float(min(np.min(est[:, iy]), np.min(gt[:, iy])))
    y_max = float(max(np.max(est[:, iy]), np.max(gt[:, iy])))
    margin = 0.1 * max(x_max - x_min, y_max - y_min, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.axis("equal")

    gt_line, = ax.plot([], [], color="black", linewidth=2.0, label="Ground Truth")
    est_line, = ax.plot([], [], color="tab:blue", linewidth=1.8, label="Estimate")
    gt_dot, = ax.plot([], [], "o", color="black", markersize=5)
    est_dot, = ax.plot([], [], "o", color="tab:blue", markersize=5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")

    gps_scatter = None
    gps = None
    if gps_measurements is not None:
        gps = np.asarray(gps_measurements, dtype=float)
        if gps.shape[0] == mask.shape[0]:
            gps = gps[mask]
        if gps.ndim == 2 and gps.shape[1] >= 2:
            gps_scatter = ax.scatter([], [], s=10, alpha=0.35, color="tab:orange", label="GPS")

    ax.legend(loc="best")

    def _update(frame: int):
        """
        Goal:
            animation의 현재 frame에 맞춰 line, marker, text artist를 갱신한다.
        Input:
            frame은 현재 animation frame index이다.
        Output:
            blit에 사용할 matplotlib artist list를 반환한다.
        """
        start = max(0, frame - int(tail_length))
        gt_line.set_data(gt[start : frame + 1, ix], gt[start : frame + 1, iy])
        est_line.set_data(est[start : frame + 1, ix], est[start : frame + 1, iy])
        gt_dot.set_data([gt[frame, ix]], [gt[frame, iy]])
        est_dot.set_data([est[frame, ix]], [est[frame, iy]])
        time_text.set_text(f"t = {ts[frame]:.2f} s")
        artists = [gt_line, est_line, gt_dot, est_dot, time_text]

        if gps_scatter is not None and gps is not None:
            gps_window = gps[start : frame + 1, :2]
            gps_scatter.set_offsets(gps_window)
            artists.append(gps_scatter)
        return artists

    anim = FuncAnimation(fig, _update, frames=ts.size, interval=1_000.0 / max(fps, 1), blit=True)

    saved_path = path
    if path.suffix.lower() == ".gif":
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        try:
            anim.save(path, fps=fps)
        except Exception:
            saved_path = path.with_suffix(".gif")
            anim.save(saved_path, writer=PillowWriter(fps=fps))

    plt.close(fig)
    return saved_path
