from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def write_metric_plots(output_dir: Path, results: list[Any]) -> list[str]:
    plots = ["rmse_position.png", "error_variance.png", "runtime_sec.png"]
    _plot_metric_bars(output_dir / plots[0], results, "rmse_position", "Position RMSE [m]")
    _plot_metric_bars(output_dir / plots[1], results, "error_variance", "Position Error Variance [m^2]")
    _plot_metric_bars(output_dir / plots[2], results, "runtime_sec", "Total Runtime [s]")
    return plots


def write_trajectory_animation(
    output_dir: Path,
    results: list[Any],
    gt: np.ndarray,
    algorithm: str = "inekf",
    max_frames: int = 600,
    fps: int = 30,
    covariance_sigma: float = 2.0,
    draw_covariances: bool = True,
) -> dict[str, dict[str, str]]:
    path = output_dir / f"trajectory_comparison_{algorithm}.mp4"
    status = _plot_trajectory_animation_3d(
        path,
        results,
        gt,
        algorithm,
        max_frames=max_frames,
        fps=fps,
        covariance_sigma=covariance_sigma,
        draw_covariances=draw_covariances,
    )
    return {algorithm: {"path": str(path), "status": status}}


def _plot_metric_bars(path: Path, results: list[Any], field: str, ylabel: str) -> None:
    ordered = sorted(results, key=lambda r: (getattr(r, "family", "") != "ours", getattr(r, "implementation", "")))
    labels = [getattr(result, "implementation", "") for result in ordered]
    values = []
    colors = []
    for idx, result in enumerate(ordered):
        value = getattr(result, field)
        values.append(np.nan if value is None else float(value))
        colors.append(_series_color(idx))

    x = np.arange(len(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(labels)), 4.0))
    bars = ax.bar(x, values, color=colors, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        label = "nan" if not np.isfinite(value) else f"{value:.4g}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() if np.isfinite(value) else 0.0, label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trajectory_animation_3d(
    path: Path,
    results: list[Any],
    gt: np.ndarray,
    algorithm: str,
    max_frames: int = 600,
    fps: int = 30,
    covariance_sigma: float = 2.0,
    draw_covariances: bool = True,
) -> str:
    ok_results = [
        result
        for result in results
        if getattr(result, "status", "") == "ok"
        and getattr(result, "algorithm", "") == algorithm
        and getattr(result, "estimates", None) is not None
    ]
    if not ok_results or len(gt) == 0:
        return f"skipped: no successful {algorithm} estimates to animate"
    if not animation.writers.is_available("ffmpeg"):
        return "skipped: ffmpeg writer is not available for mp4 output"

    finite_results = []
    for result in ok_results:
        estimates = np.asarray(result.estimates, dtype=float)
        if estimates.ndim == 2 and estimates.shape[1] >= 3 and np.any(np.all(np.isfinite(estimates[:, :3]), axis=1)):
            finite_results.append(result)
    if not finite_results:
        return "skipped: no finite estimate positions to animate"

    n_steps = min([len(gt), *[len(result.estimates) for result in finite_results]])
    if n_steps == 0:
        return "skipped: empty estimate arrays"
    frame_count = max(1, min(int(max_frames), n_steps))
    frame_indices = np.linspace(0, n_steps - 1, frame_count, dtype=int)

    gt_xyz = np.asarray(gt[:n_steps, :3], dtype=float)
    finite_gt = gt_xyz[np.all(np.isfinite(gt_xyz), axis=1)]
    if finite_gt.size == 0:
        return "skipped: no finite trajectory points"

    mins = finite_gt.min(axis=0)
    maxs = finite_gt.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1.0)
    limits = [(center[i] - 0.75 * span, center[i] + 0.75 * span) for i in range(3)]
    plot_radius = 2.5 * span
    covariance_min_radius = max(0.015 * span, 0.03)

    errors_by_label = {}
    for result in finite_results:
        estimates = np.asarray(result.estimates[:n_steps, :3], dtype=float)
        label = f"{getattr(result, 'implementation', 'estimate')} {algorithm}"
        errors_by_label[label] = np.linalg.norm(estimates - gt_xyz, axis=1)
    max_error = max([float(np.nanmax(errors)) for errors in errors_by_label.values()] + [1.0])

    fig = plt.figure(figsize=(13, 5.8))
    ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
    ax_err = fig.add_subplot(1, 2, 2)
    ax_traj.set_xlim(*limits[0])
    ax_traj.set_ylim(*limits[1])
    ax_traj.set_zlim(*limits[2])
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_zlabel("z [m]")
    ax_traj.set_title(f"{algorithm.upper()} 3D Trajectory")
    ax_traj.grid(alpha=0.25)

    ax_err.set_title("Position Error")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("error [m]")
    ax_err.set_xlim(0, n_steps - 1)
    ax_err.set_ylim(0, max_error * 1.08)
    ax_err.grid(alpha=0.25)

    gt_line, = ax_traj.plot([], [], [], color="#111111", linewidth=2.4, label="GT")
    gt_point, = ax_traj.plot([], [], [], marker="o", color="#111111", markersize=4)
    estimate_lines = []
    estimate_points = []
    error_lines = []
    covariance_sources = []
    for idx, result in enumerate(finite_results):
        estimates = np.asarray(result.estimates[:n_steps, :3], dtype=float)
        estimates = _mask_outlier_points(estimates, center, plot_radius)
        label = f"{getattr(result, 'implementation', 'estimate')} {algorithm}"
        color = _series_color(idx)
        line, = ax_traj.plot([], [], [], color=color, linewidth=2.0, label=label)
        point, = ax_traj.plot([], [], [], marker="o", color=color, markersize=3)
        rmse = float(np.sqrt(np.nanmean(errors_by_label[label] ** 2)))
        err_line, = ax_err.plot([], [], color=color, linewidth=2.0, label=f"{label} RMSE={rmse:.3f} m")
        estimate_lines.append((line, estimates))
        estimate_points.append((point, estimates))
        error_lines.append((err_line, errors_by_label[label]))
        covariances = getattr(result, "covariances", None)
        if draw_covariances and covariances is not None:
            covariances = np.asarray(covariances, dtype=float)
            if covariances.ndim == 3 and covariances.shape[1:] == (3, 3):
                covariance_sources.append(
                    (
                        estimates,
                        _display_covariance_sequence(covariances[: min(n_steps, len(covariances)), :3, :3]),
                        color,
                        0.14,
                        idx,
                    )
                )
    ax_traj.legend(loc="best", fontsize=8)
    ax_err.legend(loc="best", fontsize=8)
    fig.tight_layout()
    covariance_surfaces: list[Poly3DCollection] = []

    def update(frame_idx: int):
        end = int(frame_idx) + 1
        x_values = np.arange(end)
        while covariance_surfaces:
            covariance_surfaces.pop().remove()
        gt_line.set_data(gt_xyz[:end, 0], gt_xyz[:end, 1])
        gt_line.set_3d_properties(gt_xyz[:end, 2])
        gt_point.set_data([gt_xyz[end - 1, 0]], [gt_xyz[end - 1, 1]])
        gt_point.set_3d_properties([gt_xyz[end - 1, 2]])
        artists = [gt_line, gt_point]
        for line, estimates in estimate_lines:
            line.set_data(estimates[:end, 0], estimates[:end, 1])
            line.set_3d_properties(estimates[:end, 2])
            artists.append(line)
        for point, estimates in estimate_points:
            point.set_data([estimates[end - 1, 0]], [estimates[end - 1, 1]])
            point.set_3d_properties([estimates[end - 1, 2]])
            artists.append(point)
        for err_line, errors in error_lines:
            err_line.set_data(x_values, errors[:end])
            artists.append(err_line)
        cov_idx = end - 1
        for xyz, covariances, color, ellipsoid_alpha, zorder in covariance_sources:
            if cov_idx >= len(covariances) or cov_idx >= len(xyz):
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
                zorder=max(1, zorder),
            )
            covariance_surfaces.append(surface)
            artists.append(surface)
        return artists

    path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / max(1, int(fps)), blit=False)
    try:
        writer = animation.FFMpegWriter(fps=max(1, int(fps)), bitrate=1800)
        anim.save(path, writer=writer, dpi=140)
    finally:
        plt.close(fig)
    return "ok"


def _mask_outlier_points(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    masked = np.asarray(points, dtype=float).copy()
    finite = np.all(np.isfinite(masked), axis=1)
    distance = np.linalg.norm(masked - center.reshape(1, 3), axis=1)
    keep = finite & (distance <= radius)
    masked[~keep, :] = np.nan
    return masked


def _covariance_ellipsoid_mesh(
    center: np.ndarray,
    covariance: np.ndarray,
    sigma: float = 2.0,
    min_radius: float = 0.0,
    resolution: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    center = np.asarray(center, dtype=float).reshape(3)
    covariance = np.asarray(covariance, dtype=float).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
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


def _display_covariance_sequence(covariances: np.ndarray) -> np.ndarray:
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        return covariances
    smoothed = np.empty_like(covariances)
    previous: np.ndarray | None = None
    alpha = 0.85
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


def _series_color(index: int) -> str:
    palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    return palette[index % len(palette)]
