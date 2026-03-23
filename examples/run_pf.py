from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# examples/에서 실행해도 루트 모듈 import가 되도록 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.particle_filter import ParticleFilter


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_imu_sequence(dataset_cfg: dict, pose_type: str) -> tuple[list[dict], np.ndarray]:
    """
    TODO: 실제 IMU/GPS 로더로 교체.
    지금은 PF 파이프라인 테스트용 시퀀스를 생성.
    """
    n = int(dataset_cfg.get("sequence_length", 300))
    dt = float(dataset_cfg.get("dt", 0.1))
    seed = int(dataset_cfg.get("seed", 10))
    mode = dataset_cfg.get("mode", "fused")

    rng = np.random.default_rng(seed)

    if pose_type == "2d":
        gt = np.zeros((n, 3), dtype=float)  # [x, y, yaw]
        controls = np.zeros((n, 2), dtype=float)  # [speed, yaw_rate]
        controls[:, 0] = 1.0 + 0.2 * np.sin(np.linspace(0.0, 6.0, n))
        controls[:, 1] = 0.12 * np.sin(np.linspace(0.0, 4.0, n))

        for k in range(1, n):
            x, y, yaw = gt[k - 1]
            v, yaw_rate = controls[k]
            yaw = yaw + yaw_rate * dt
            x = x + v * np.cos(yaw) * dt
            y = y + v * np.sin(yaw) * dt
            gt[k] = np.array([x, y, yaw], dtype=float)

        meas_std = np.array(dataset_cfg.get("gps_noise_std", [0.7, 0.7]), dtype=float)
        measurements = gt[:, :2] + rng.normal(0.0, meas_std, size=(n, 2))

    else:
        gt = np.zeros((n, 6), dtype=float)  # [x, y, z, roll, pitch, yaw]
        controls = np.zeros((n, 6), dtype=float)  # 단순 additive control
        controls[:, 0] = 0.8 + 0.2 * np.sin(np.linspace(0.0, 5.0, n))
        controls[:, 1] = 0.3 * np.cos(np.linspace(0.0, 3.0, n))
        controls[:, 2] = 0.1 * np.sin(np.linspace(0.0, 2.0, n))
        controls[:, 5] = 0.08 * np.sin(np.linspace(0.0, 4.0, n))

        for k in range(1, n):
            gt[k] = gt[k - 1] + controls[k] * dt

        meas_std = np.array(dataset_cfg.get("gps_noise_std", [0.7, 0.7, 0.7]), dtype=float)
        if meas_std.size == 2:
            meas_std = np.array([meas_std[0], meas_std[1], 0.7], dtype=float)
        measurements = gt[:, :3] + rng.normal(0.0, meas_std[:3], size=(n, 3))

    # mode에 맞게 measurement 입력 제어
    dataset = []
    for k in range(n):
        measurement = None if mode == "imu_only" else measurements[k]
        dataset.append(
            {
                "control": controls[k],
                "measurement": measurement,
                "dt": dt,
                "gt": gt[k],
            }
        )

    return dataset, gt


def compute_rmse(estimates: np.ndarray, gt: np.ndarray, pose_type: str) -> float:
    pos_dim = 2 if pose_type == "2d" else 3
    err = estimates[:, :pos_dim] - gt[:, :pos_dim]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def plot_results(estimates: np.ndarray, gt: np.ndarray, pose_type: str, save_path: Path) -> None:
    if pose_type == "2d":
        pos_err = np.linalg.norm(estimates[:, :2] - gt[:, :2], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(gt[:, 0], gt[:, 1], label="GT", linewidth=2.0)
        axes[0].plot(estimates[:, 0], estimates[:, 1], label="PF", linewidth=1.5)
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
        fig = plt.figure(figsize=(14, 5))
        ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
        ax_err = fig.add_subplot(1, 2, 2)

        ax_traj.plot(gt[:, 0], gt[:, 1], gt[:, 2], label="GT", linewidth=2.0)
        ax_traj.plot(estimates[:, 0], estimates[:, 1], estimates[:, 2], label="PF", linewidth=1.5)
        ax_traj.set_title("3D Trajectory")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.set_zlabel("z")
        ax_traj.legend()

        ax_err.plot(pos_err, color="tab:red", linewidth=1.5)
        ax_err.set_title("Position Error Norm")
        ax_err.set_xlabel("step")
        ax_err.set_ylabel("error [m]")
        ax_err.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Particle Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--pf-config", default=str(PROJECT_ROOT / "config" / "pf.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    dataset_cfg = load_yaml(Path(args.dataset_config))
    pf_cfg = load_yaml(Path(args.pf_config))

    pose_type = dataset_cfg.get("pose_type", "2d")
    if pose_type == "6d":
        pose_type = "3d"

    pf = ParticleFilter.from_configs(dataset_cfg, pf_cfg)

    dataset, gt = make_imu_sequence(dataset_cfg, pose_type=pose_type)
    estimates = pf.run(dataset)

    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[PF] Pose type: {pose_type}")
    print(f"[PF] Steps: {len(dataset)}")
    print(f"[PF] RMSE (position): {rmse:.4f}")

    out_dir = Path(args.output_dir)
    plot_path = out_dir / f"pf_trajectory_{pose_type}.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path)
    print(f"[PF] Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
