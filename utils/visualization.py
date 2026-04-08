from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def plot_results(
    estimates: np.ndarray,
    gt: np.ndarray,
    pose_type: str,
    save_path: Path,
    visual_cfg: dict | None = None,
) -> None:
    visual_cfg = visual_cfg or {}
    estimator_label = str(visual_cfg.get("estimator_label", "PF"))
    dpi = 150
    linewidth_gt = 1.8 if pose_type == "3d" else 2.0
    linewidth_est = 1.0 if pose_type == "3d" else 1.5
    alpha_gt = 0.9 if pose_type == "3d" else 1.0
    alpha_est = 0.75 if pose_type == "3d" else 1.0
    start_marker_size = 24.0
    end_marker_size = 32.0

    if pose_type == "2d":
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        ax.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=linewidth_gt, alpha=alpha_gt)
        ax.plot(estimates[:, 0], estimates[:, 1], label=estimator_label, linewidth=linewidth_est, alpha=alpha_est)
        ax.scatter(gt[0, 0], gt[0, 1], s=start_marker_size, color="tab:blue", marker="o", label="Start")
        ax.scatter(gt[-1, 0], gt[-1, 1], s=end_marker_size, color="tab:blue", marker="x", label="End")
        ax.set_title("2D Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
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
            fig = plt.figure(figsize=(7, 5.5))
            ax_traj = fig.add_subplot(1, 1, 1, projection="3d")

        ax_traj.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], label="GT", linewidth=linewidth_gt, alpha=alpha_gt)
        ax_traj.plot(est_plot[:, 0], est_plot[:, 1], est_plot[:, 2], label=estimator_label, linewidth=linewidth_est, alpha=alpha_est)
        ax_traj.scatter(gt_plot[0, 0], gt_plot[0, 1], gt_plot[0, 2], s=start_marker_size, color="tab:blue", marker="o")
        ax_traj.scatter(gt_plot[-1, 0], gt_plot[-1, 1], gt_plot[-1, 2], s=end_marker_size, color="tab:blue", marker="x")
        ax_traj.set_title(f"3D Trajectory (every {stride}th sample)")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.set_zlabel("z")
        ax_traj.view_init(elev=elev, azim=azim)
        ax_traj.legend()

        if show_projections:
            _plot_projection(ax_xy, gt_plot[:, 0], gt_plot[:, 1], est_plot[:, 0], est_plot[:, 1], "XY Projection", "x", "y", estimator_label)
            _plot_projection(ax_xz, gt_plot[:, 0], gt_plot[:, 2], est_plot[:, 0], est_plot[:, 2], "XZ Projection", "x", "z", estimator_label)
            _plot_projection(ax_yz, gt_plot[:, 1], gt_plot[:, 2], est_plot[:, 1], est_plot[:, 2], "YZ Projection", "y", "z", estimator_label)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)




def control_trajectory_from_dataset(dataset: list[dict], gt: np.ndarray, pose_type: str) -> np.ndarray | None:
    if gt.size == 0:
        return gt.copy()

    trajectory = gt.copy()
    state = gt[0].copy()
    trajectory[0] = state

    for idx in range(1, len(dataset)):
        sample = dataset[idx]
        control = sample.get("raw_control", sample.get("control"))
        if control is None:
            return None

        control_vec = np.asarray(control, dtype=float).reshape(-1)
        dt = float(sample.get("dt", 1.0))
        state = state.copy()

        if pose_type == "2d":
            if control_vec.size < 2:
                return None
            speed, yaw_rate = control_vec[0], control_vec[1]
            state[0] += speed * np.cos(state[2]) * dt
            state[1] += speed * np.sin(state[2]) * dt
            state[2] = np.arctan2(np.sin(state[2] + yaw_rate * dt), np.cos(state[2] + yaw_rate * dt))
        else:
            if control_vec.size < 6:
                return None
            state[:6] += control_vec[:6] * dt
            state[3:6] = np.arctan2(np.sin(state[3:6]), np.cos(state[3:6]))

        trajectory[idx] = state

    return trajectory


def measurement_trajectory_from_dataset(dataset: list[dict], gt: np.ndarray, pose_type: str) -> np.ndarray | None:
    if gt.size == 0:
        return gt.copy()

    pos_dim = 2 if pose_type == "2d" else 3
    measurements = []
    for sample in dataset:
        measurement = sample.get("raw_measurement", sample.get("measurement"))
        if measurement is None:
            return None
        measurement_vec = np.asarray(measurement, dtype=float).reshape(-1)
        if measurement_vec.size < pos_dim:
            return None
        measurements.append(measurement_vec[:pos_dim])

    if not measurements:
        return np.zeros((0, gt.shape[1]), dtype=float)

    trajectory = gt.copy()
    measurement_positions = np.vstack(measurements)
    trajectory[: measurement_positions.shape[0], :pos_dim] = measurement_positions
    return trajectory

def plot_position_error_norm(
    estimates: np.ndarray,
    gt: np.ndarray,
    pose_type: str,
    save_path: Path,
    visual_cfg: dict | None = None,
) -> None:
    visual_cfg = visual_cfg or {}
    pos_dim = 2 if pose_type == "2d" else 3
    pos_err = np.linalg.norm(estimates[:, :pos_dim] - gt[:, :pos_dim], axis=1)
    moving_avg = _moving_average(pos_err, int(visual_cfg.get("error_moving_average_window", 50)))
    median_err = float(np.median(pos_err)) if pos_err.size else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
    ax.plot(pos_err, color="tab:red", linewidth=1.2, alpha=0.8, label="Position error norm")
    ax.plot(moving_avg, color="black", linewidth=2.0, alpha=0.95, label="Moving average")
    ax.set_title(f"Position Error Norm (median: {median_err:.4f} m)")
    ax.set_xlabel("step")
    ax.set_ylabel("error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
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
    estimator_label: str,
) -> None:
    ax.plot(gt_x, gt_y, color="tab:blue", linewidth=1.4, alpha=0.9, label="GT")
    ax.plot(est_x, est_y, color="tab:orange", linewidth=0.9, alpha=0.9, label=estimator_label)
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
    moving_average_window: int = 50,
    estimator_label: str = "PF",
) -> bool:
    if estimates.size == 0 or gt.size == 0:
        return False

    suffix = save_path.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError("save_path must use .mp4 or .gif for animation export.")
    if suffix == ".mp4" and not animation.writers.is_available("ffmpeg"):
        return False
    if suffix == ".gif" and not animation.writers.is_available("pillow"):
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames = len(estimates)
    tail_length = max(1, int(tail_length))
    fps = max(1, int(fps))
    moving_average_window = max(1, int(moving_average_window))

    if pose_type == "2d":
        pos_err = np.linalg.norm(estimates[:, :2] - gt[:, :2], axis=1)
        moving_avg = _moving_average(pos_err, moving_average_window)
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
        ax_err.set_ylim(0.0, max(1e-6, float(max(np.max(pos_err), np.max(moving_avg))) * 1.1))

        ax_traj.plot(gt[:, 0], gt[:, 1], color="tab:blue", alpha=0.25, linewidth=2.0, label="GT full")
        gt_line, = ax_traj.plot([], [], color="tab:blue", linewidth=2.0, label="GT")
        est_line, = ax_traj.plot([], [], color="tab:orange", linewidth=1.5, label=estimator_label)
        gt_point, = ax_traj.plot([], [], marker="o", color="tab:blue")
        est_point, = ax_traj.plot([], [], marker="o", color="tab:orange")
        err_line, = ax_err.plot([], [], color="tab:red", linewidth=1.5, label="Error norm")
        avg_line, = ax_err.plot([], [], color="black", linewidth=2.0, label="Moving average")
        ax_traj.legend()
        ax_err.legend()

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
            avg_line.set_data(np.arange(frame + 1), moving_avg[: frame + 1])
            return gt_line, est_line, gt_point, est_point, err_line, avg_line

    else:
        pos_err = np.linalg.norm(estimates[:, :3] - gt[:, :3], axis=1)
        moving_avg = _moving_average(pos_err, moving_average_window)
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
        ax_err.set_ylim(0.0, max(1e-6, float(max(np.max(pos_err), np.max(moving_avg))) * 1.1))

        ax_traj.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="tab:blue", alpha=0.25, linewidth=2.0, label="GT full")
        gt_line, = ax_traj.plot([], [], [], color="tab:blue", linewidth=2.0, label="GT")
        est_line, = ax_traj.plot([], [], [], color="tab:orange", linewidth=1.5, label=estimator_label)
        gt_point, = ax_traj.plot([], [], [], marker="o", color="tab:blue")
        est_point, = ax_traj.plot([], [], [], marker="o", color="tab:orange")
        err_line, = ax_err.plot([], [], color="tab:red", linewidth=1.5, label="Error norm")
        avg_line, = ax_err.plot([], [], color="black", linewidth=2.0, label="Moving average")
        ax_traj.legend()
        ax_err.legend()

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
            avg_line.set_data(np.arange(frame + 1), moving_avg[: frame + 1])
            return gt_line, est_line, gt_point, est_point, err_line, avg_line

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    else:
        writer = animation.PillowWriter(fps=fps)

    progress_bar = None

    def progress_callback(frame_idx: int, total_frames: int) -> None:
        nonlocal progress_bar
        if tqdm is None:
            return
        if progress_bar is None:
            total = total_frames if total_frames and total_frames > 0 else frames
            progress_bar = tqdm(total=total, desc=f"Saving {save_path.name}", unit="frame")
        target_n = min(frame_idx + 1, progress_bar.total)
        delta = target_n - progress_bar.n
        if delta > 0:
            progress_bar.update(delta)

    try:
        ani.save(save_path, writer=writer, progress_callback=progress_callback)
    finally:
        if progress_bar is not None:
            if progress_bar.n < progress_bar.total:
                progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()
        plt.close(fig)
    return True


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window, dtype=float) / float(window)
    averaged = np.convolve(values, kernel, mode="valid")
    if window == 1:
        return averaged
    prefix = np.empty(window - 1, dtype=float)
    cumsum = np.cumsum(values[: window - 1], dtype=float)
    prefix[:] = cumsum / np.arange(1, window, dtype=float)
    return np.concatenate([prefix, averaged])
