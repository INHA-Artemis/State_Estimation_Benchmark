from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def load_synthetic_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gt_points: list[list[float]] = []
    noisy_points: list[list[float]] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gt_x", "gt_y", "gnss_x", "gnss_y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            gt_points.append([float(row["gt_x"]), float(row["gt_y"])])
            noisy_points.append([float(row["gnss_x"]), float(row["gnss_y"])])

    if not gt_points:
        raise ValueError(f"No rows found in {csv_path}")

    return np.asarray(gt_points, dtype=float), np.asarray(noisy_points, dtype=float)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    coeff = 1.0 / (std * np.sqrt(2.0 * np.pi))
    z = (x - float(mean)) / std
    return coeff * np.exp(-0.5 * z * z)


def fit_two_component_gmm_1d(values: np.ndarray, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 1 or values.size < 4:
        raise ValueError("Need a 1D array with at least 4 samples for GMM fitting.")

    q25, q75 = np.quantile(values, [0.25, 0.75])
    means = np.asarray([q25, q75], dtype=float)
    std_init = max(float(np.std(values)), 1e-3)
    stds = np.asarray([std_init, std_init], dtype=float)
    weights = np.asarray([0.5, 0.5], dtype=float)

    for _ in range(max_iter):
        pdf0 = weights[0] * normal_pdf(values, means[0], stds[0])
        pdf1 = weights[1] * normal_pdf(values, means[1], stds[1])
        total = np.clip(pdf0 + pdf1, 1e-12, None)

        resp0 = pdf0 / total
        resp1 = pdf1 / total
        responsibilities = np.vstack([resp0, resp1])

        nk = np.clip(responsibilities.sum(axis=1), 1e-8, None)
        new_weights = nk / values.size
        new_means = (responsibilities @ values) / nk
        new_stds = np.sqrt(np.clip((responsibilities * ((values[None, :] - new_means[:, None]) ** 2)).sum(axis=1) / nk, 1e-8, None))

        if (
            np.allclose(weights, new_weights, atol=1e-6)
            and np.allclose(means, new_means, atol=1e-6)
            and np.allclose(stds, new_stds, atol=1e-6)
        ):
            weights, means, stds = new_weights, new_means, new_stds
            break

        weights, means, stds = new_weights, new_means, new_stds

    order = np.argsort(means)
    return weights[order], means[order], stds[order]


def gmm_pdf(x: np.ndarray, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    mixture = np.zeros_like(x, dtype=float)
    for weight, mean, std in zip(weights, means, stds):
        mixture += float(weight) * normal_pdf(x, float(mean), float(std))
    return mixture


def build_cov_ellipse(points: np.ndarray, ax: plt.Axes, color: str, label: str) -> None:
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.clip(vals, 1e-12, None))
    center = points.mean(axis=0)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        linewidth=2.0,
        linestyle="--",
        label=label,
    )
    ax.add_patch(ellipse)


def plot_trajectory_and_noise(gt: np.ndarray, noisy: np.ndarray, save_path: Path) -> dict[str, float]:
    residuals = noisy - gt
    x_err = residuals[:, 0]
    y_err = residuals[:, 1]

    x_mean = float(np.mean(x_err))
    y_mean = float(np.mean(y_err))
    x_std = float(np.std(x_err))
    y_std = float(np.std(y_err))
    pos_rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    gmm_x = fit_two_component_gmm_1d(x_err)
    gmm_y = fit_two_component_gmm_1d(y_err)

    x_axis = np.linspace(x_err.min() - 0.5, x_err.max() + 0.5, 600)
    y_axis = np.linspace(y_err.min() - 0.5, y_err.max() + 0.5, 600)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_traj, ax_scatter, ax_x, ax_y = axes.flat

    ax_traj.plot(gt[:, 0], gt[:, 1], color="tab:blue", linewidth=2.2, label="GT trajectory")
    ax_traj.plot(noisy[:, 0], noisy[:, 1], color="tab:orange", linewidth=1.2, alpha=0.85, label="Noisy trajectory")
    ax_traj.scatter(gt[0, 0], gt[0, 1], color="tab:blue", s=40, marker="o", label="Start")
    ax_traj.scatter(gt[-1, 0], gt[-1, 1], color="tab:blue", s=55, marker="x", label="End")
    ax_traj.set_title("GT vs Noisy Trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()

    ax_scatter.scatter(x_err, y_err, s=18, alpha=0.35, color="tab:red", label="GT -> noisy residual")
    ax_scatter.scatter([0.0], [0.0], color="black", s=55, marker="x", label="GT-centered mean")
    build_cov_ellipse(residuals, ax_scatter, color="tab:green", label="Single Gaussian 1-sigma")
    ax_scatter.set_title("Residual Cloud in Measurement Space")
    ax_scatter.set_xlabel("gnss_x - gt_x [m]")
    ax_scatter.set_ylabel("gnss_y - gt_y [m]")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()

    ax_x.hist(x_err, bins=32, density=True, alpha=0.35, color="tab:orange", label="Empirical histogram")
    ax_x.plot(x_axis, normal_pdf(x_axis, x_mean, x_std), color="tab:green", linewidth=2.2, label="Single Gaussian fit")
    ax_x.plot(x_axis, gmm_pdf(x_axis, *gmm_x), color="tab:red", linewidth=2.2, label="2-comp GMM fit")
    ax_x.set_title("X-axis Noise Distribution")
    ax_x.set_xlabel("gnss_x - gt_x [m]")
    ax_x.set_ylabel("density")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend()

    ax_y.hist(y_err, bins=32, density=True, alpha=0.35, color="tab:orange", label="Empirical histogram")
    ax_y.plot(y_axis, normal_pdf(y_axis, y_mean, y_std), color="tab:green", linewidth=2.2, label="Single Gaussian fit")
    ax_y.plot(y_axis, gmm_pdf(y_axis, *gmm_y), color="tab:red", linewidth=2.2, label="2-comp GMM fit")
    ax_y.set_title("Y-axis Noise Distribution")
    ax_y.set_xlabel("gnss_y - gt_y [m]")
    ax_y.set_ylabel("density")
    ax_y.grid(True, alpha=0.3)
    ax_y.legend()

    fig.suptitle(
        "Synthetic trajectory and GT-to-noisy distribution shift\n"
        "Green: single Gaussian approximation, Red: 2-component GMM approximation",
        fontsize=14,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "samples": float(len(gt)),
        "rmse": pos_rmse,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "gmm_x_weight_0": float(gmm_x[0][0]),
        "gmm_x_mean_0": float(gmm_x[1][0]),
        "gmm_x_std_0": float(gmm_x[2][0]),
        "gmm_x_weight_1": float(gmm_x[0][1]),
        "gmm_x_mean_1": float(gmm_x[1][1]),
        "gmm_x_std_1": float(gmm_x[2][1]),
        "gmm_y_weight_0": float(gmm_y[0][0]),
        "gmm_y_mean_0": float(gmm_y[1][0]),
        "gmm_y_std_0": float(gmm_y[2][0]),
        "gmm_y_weight_1": float(gmm_y[0][1]),
        "gmm_y_mean_1": float(gmm_y[1][1]),
        "gmm_y_std_1": float(gmm_y[2][1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot synthetic GT/noisy trajectory and compare empirical noise against Gaussian/GMM fits."
    )
    parser.add_argument(
        "--csv-path",
        default="/workspace/State_Estimation_Benchmark/outputs/synthetic_test.csv",
        help="Synthetic CSV file containing gt_x, gt_y, gnss_x, gnss_y columns.",
    )
    parser.add_argument(
        "--save-path",
        default="",
        help="Output image path. Defaults to outputs/<csv_stem>_trajectory_noise.png",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.save_path:
        save_path = Path(args.save_path).resolve()
    else:
        save_path = csv_path.parent / f"{csv_path.stem}_trajectory_noise.png"

    gt, noisy = load_synthetic_csv(csv_path)
    summary = plot_trajectory_and_noise(gt, noisy, save_path=save_path)

    print(f"[Plot] CSV: {csv_path}")
    print(f"[Plot] Saved figure: {save_path}")
    print(f"[Plot] Samples: {int(summary['samples'])}")
    print(f"[Plot] Position RMSE: {summary['rmse']:.4f} m")
    print(f"[Plot] X noise mean/std: {summary['x_mean']:.4f} / {summary['x_std']:.4f}")
    print(f"[Plot] Y noise mean/std: {summary['y_mean']:.4f} / {summary['y_std']:.4f}")
    print(
        "[Plot] X GMM components: "
        f"w={summary['gmm_x_weight_0']:.3f}, mu={summary['gmm_x_mean_0']:.3f}, sigma={summary['gmm_x_std_0']:.3f} | "
        f"w={summary['gmm_x_weight_1']:.3f}, mu={summary['gmm_x_mean_1']:.3f}, sigma={summary['gmm_x_std_1']:.3f}"
    )
    print(
        "[Plot] Y GMM components: "
        f"w={summary['gmm_y_weight_0']:.3f}, mu={summary['gmm_y_mean_0']:.3f}, sigma={summary['gmm_y_std_0']:.3f} | "
        f"w={summary['gmm_y_weight_1']:.3f}, mu={summary['gmm_y_mean_1']:.3f}, sigma={summary['gmm_y_std_1']:.3f}"
    )


if __name__ == "__main__":
    main()
