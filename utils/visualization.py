from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_results(
    estimates: np.ndarray,
    gt: np.ndarray,
    pose_type: str,
    save_path: Path,
    visual_cfg: dict | None = None,
) -> None:
    visual_cfg = visual_cfg or {}
    dpi = 150
    linewidth_gt = 1.8 if pose_type == "3d" else 2.0
    linewidth_pf = 1.0 if pose_type == "3d" else 1.5
    alpha_gt = 0.9 if pose_type == "3d" else 1.0
    alpha_pf = 0.75 if pose_type == "3d" else 1.0
    start_marker_size = 24.0
    end_marker_size = 32.0

    if pose_type == "2d":
        pos_err = np.linalg.norm(estimates[:, :2] - gt[:, :2], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(gt[:, 0], gt[:, 1], label="GT", linewidth=linewidth_gt, alpha=alpha_gt)
        axes[0].plot(estimates[:, 0], estimates[:, 1], label="PF", linewidth=linewidth_pf, alpha=alpha_pf)
        axes[0].scatter(gt[0, 0], gt[0, 1], s=start_marker_size, color="tab:blue", marker="o", label="Start")
        axes[0].scatter(gt[-1, 0], gt[-1, 1], s=end_marker_size, color="tab:blue", marker="x", label="End")
        axes[0].set_title("2D Trajectory")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(pos_err, color="tab:red", linewidth=1.5)
        axes[1].set_title("Position Error Norm")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("error [m]")
        axes[1].grid(True, alpha=0.3)
    else:
        pos_err = np.linalg.norm(estimates[:, :3] - gt[:, :3], axis=1)
        stride = max(1, int(visual_cfg.get("downsample", 1)))
        elev = float(visual_cfg.get("elev", 20.0))
        azim = float(visual_cfg.get("azim", 35.0))
        show_projections = bool(visual_cfg.get("show_projections", True))

        gt_plot = gt[::stride]
        est_plot = estimates[::stride]

        if show_projections:
            fig = plt.figure(figsize=(14, 10))
            ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
            ax_xy = fig.add_subplot(2, 2, 2)
            ax_xz = fig.add_subplot(2, 2, 3)
            ax_yz = fig.add_subplot(2, 2, 4)
        else:
            fig = plt.figure(figsize=(14, 5))
            ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
            ax_err = fig.add_subplot(1, 2, 2)

        ax_traj.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], label="GT", linewidth=linewidth_gt, alpha=alpha_gt)
        ax_traj.plot(est_plot[:, 0], est_plot[:, 1], est_plot[:, 2], label="PF", linewidth=linewidth_pf, alpha=alpha_pf)
        ax_traj.scatter(gt_plot[0, 0], gt_plot[0, 1], gt_plot[0, 2], s=start_marker_size, color="tab:blue", marker="o")
        ax_traj.scatter(gt_plot[-1, 0], gt_plot[-1, 1], gt_plot[-1, 2], s=end_marker_size, color="tab:blue", marker="x")
        ax_traj.set_title(f"3D Trajectory (every {stride}th sample)")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.set_zlabel("z")
        ax_traj.view_init(elev=elev, azim=azim)
        ax_traj.legend()

        if show_projections:
            _plot_projection(ax_xy, gt_plot[:, 0], gt_plot[:, 1], est_plot[:, 0], est_plot[:, 1], "XY Projection", "x", "y")
            _plot_projection(ax_xz, gt_plot[:, 0], gt_plot[:, 2], est_plot[:, 0], est_plot[:, 2], "XZ Projection", "x", "z")
            _plot_projection(ax_yz, gt_plot[:, 1], gt_plot[:, 2], est_plot[:, 1], est_plot[:, 2], "YZ Projection", "y", "z")
        else:
            ax_err.plot(pos_err, color="tab:red", linewidth=1.5)
            ax_err.set_title("Position Error Norm")
            ax_err.set_xlabel("step")
            ax_err.set_ylabel("error [m]")
            ax_err.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_projection(
    ax,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.plot(gt_x, gt_y, color="tab:blue", linewidth=1.4, alpha=0.9, label="GT")
    ax.plot(est_x, est_y, color="tab:orange", linewidth=0.9, alpha=0.9, label="PF")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()


def save_trajectory_animation(
    estimates: np.ndarray,
    gt: np.ndarray,
    pose_type: str,
    save_path: Path,
    fps: int = 20,
    tail_length: int = 80,
) -> bool:
    if estimates.size == 0 or gt.size == 0:
        return False
    if save_path.suffix.lower() != ".mp4":
        raise ValueError("save_path must use .mp4 for MP4 export.")
    if not animation.writers.is_available("ffmpeg"):
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames = len(estimates)
    tail_length = max(1, int(tail_length))
    fps = max(1, int(fps))

    if pose_type == "2d":
        pos_err = np.linalg.norm(estimates[:, :2] - gt[:, :2], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_traj, ax_err = axes

        ax_traj.set_title("2D Trajectory")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.axis("equal")
        ax_traj.grid(True, alpha=0.3)

        ax_err.set_title("Position Error Norm")
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("error [m]")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_xlim(0, frames - 1)
        ax_err.set_ylim(0.0, max(1e-6, float(np.max(pos_err)) * 1.1))

        ax_traj.plot(gt[:, 0], gt[:, 1], color="tab:blue", alpha=0.25, linewidth=2.0, label="GT full")
        gt_line, = ax_traj.plot([], [], color="tab:blue", linewidth=2.0, label="GT")
        est_line, = ax_traj.plot([], [], color="tab:orange", linewidth=1.5, label="PF")
        gt_point, = ax_traj.plot([], [], marker="o", color="tab:blue")
        est_point, = ax_traj.plot([], [], marker="o", color="tab:orange")
        err_line, = ax_err.plot([], [], color="tab:red", linewidth=1.5)
        ax_traj.legend()

        margin = 0.5
        ax_traj.set_xlim(float(min(gt[:, 0].min(), estimates[:, 0].min()) - margin), float(max(gt[:, 0].max(), estimates[:, 0].max()) + margin))
        ax_traj.set_ylim(float(min(gt[:, 1].min(), estimates[:, 1].min()) - margin), float(max(gt[:, 1].max(), estimates[:, 1].max()) + margin))

        def update(frame: int):
            start = max(0, frame - tail_length + 1)
            gt_seg = gt[start : frame + 1]
            est_seg = estimates[start : frame + 1]
            gt_line.set_data(gt_seg[:, 0], gt_seg[:, 1])
            est_line.set_data(est_seg[:, 0], est_seg[:, 1])
            gt_point.set_data([gt[frame, 0]], [gt[frame, 1]])
            est_point.set_data([estimates[frame, 0]], [estimates[frame, 1]])
            err_line.set_data(np.arange(frame + 1), pos_err[: frame + 1])
            return gt_line, est_line, gt_point, est_point, err_line

    else:
        pos_err = np.linalg.norm(estimates[:, :3] - gt[:, :3], axis=1)
        fig = plt.figure(figsize=(14, 5))
        ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
        ax_err = fig.add_subplot(1, 2, 2)

        ax_traj.set_title("3D Trajectory")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.set_zlabel("z")
        ax_err.set_title("Position Error Norm")
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("error [m]")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_xlim(0, frames - 1)
        ax_err.set_ylim(0.0, max(1e-6, float(np.max(pos_err)) * 1.1))

        ax_traj.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="tab:blue", alpha=0.25, linewidth=2.0, label="GT full")
        gt_line, = ax_traj.plot([], [], [], color="tab:blue", linewidth=2.0, label="GT")
        est_line, = ax_traj.plot([], [], [], color="tab:orange", linewidth=1.5, label="PF")
        gt_point, = ax_traj.plot([], [], [], marker="o", color="tab:blue")
        est_point, = ax_traj.plot([], [], [], marker="o", color="tab:orange")
        err_line, = ax_err.plot([], [], color="tab:red", linewidth=1.5)
        ax_traj.legend()

        margin = 0.5
        ax_traj.set_xlim(float(min(gt[:, 0].min(), estimates[:, 0].min()) - margin), float(max(gt[:, 0].max(), estimates[:, 0].max()) + margin))
        ax_traj.set_ylim(float(min(gt[:, 1].min(), estimates[:, 1].min()) - margin), float(max(gt[:, 1].max(), estimates[:, 1].max()) + margin))
        ax_traj.set_zlim(float(min(gt[:, 2].min(), estimates[:, 2].min()) - margin), float(max(gt[:, 2].max(), estimates[:, 2].max()) + margin))

        def update(frame: int):
            start = max(0, frame - tail_length + 1)
            gt_seg = gt[start : frame + 1]
            est_seg = estimates[start : frame + 1]
            gt_line.set_data(gt_seg[:, 0], gt_seg[:, 1])
            gt_line.set_3d_properties(gt_seg[:, 2])
            est_line.set_data(est_seg[:, 0], est_seg[:, 1])
            est_line.set_3d_properties(est_seg[:, 2])
            gt_point.set_data([gt[frame, 0]], [gt[frame, 1]])
            gt_point.set_3d_properties([gt[frame, 2]])
            est_point.set_data([estimates[frame, 0]], [estimates[frame, 1]])
            est_point.set_3d_properties([estimates[frame, 2]])
            err_line.set_data(np.arange(frame + 1), pos_err[: frame + 1])
            return gt_line, est_line, gt_point, est_point, err_line

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig)
    return True
