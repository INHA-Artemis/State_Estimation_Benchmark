from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
FILTER_COLORS = {
    "ekf": "tab:blue",
    "pf": "tab:orange",
    "ukf": "tab:green",
    "inekf": "tab:red",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_csv(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        return []

    rows = sorted(rows, key=lambda row: _float(row, "rmse_mean"))
    labels = [f"{row['filter']} | {row['case']}" for row in rows]
    rmse = [_float(row, "rmse_mean") for row in rows]
    runtime = [_float(row, "runtime_mean") for row in rows]
    y = list(range(len(rows)))
    colors = [FILTER_COLORS.get(row["filter"], "tab:gray") for row in rows]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(11.5, max(6.0, len(rows) * 0.9)))
    bars = ax.barh(y, rmse, color=colors, alpha=0.88)
    ax.set_title("RMSE Mean by Filter Case")
    ax.set_xlabel("RMSE mean")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.01,
        1.02,
        "Lower RMSE is better. Runtime is shown as text at the end of each bar.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )

    x_max = max(rmse) if rmse else 0.0
    ax.set_xlim(0.0, x_max * 1.28)
    text_offset = max(x_max * 0.02, 0.004)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_width() + text_offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"runtime {runtime[idx]:.3f}s",
            ha="left",
            va="center",
            fontsize=9,
            color="dimgray",
        )

    summary_plot = out_dir / f"{csv_path.stem}_bars.png"
    fig.tight_layout()
    _save_figure(fig, summary_plot)
    saved.append(summary_plot)

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    legend_handles = []
    seen_filters: set[str] = set()
    for row in rows:
        filter_name = row["filter"]
        if filter_name in seen_filters:
            continue
        seen_filters.add(filter_name)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=FILTER_COLORS.get(filter_name, "tab:gray"),
                markersize=10,
                label=filter_name.upper(),
            )
        )

    point_handles = []
    for idx, row in enumerate(rows, start=1):
        color = FILTER_COLORS.get(row["filter"], "tab:gray")
        ax.scatter(runtime[idx - 1], rmse[idx - 1], s=95, color=color, alpha=0.9, edgecolor="black", linewidth=0.4)
        ax.annotate(str(idx), (runtime[idx - 1], rmse[idx - 1]), textcoords="offset points", xytext=(5, 4), fontsize=9, weight="bold")
        point_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=9,
                label=f"{idx}. {row['filter']} | {row['case']} | rmse={rmse[idx - 1]:.3f} | t={runtime[idx - 1]:.3f}s",
            )
        )

    ax.set_title("RMSE Mean vs Runtime")
    ax.set_xlabel("Runtime mean (sec)")
    ax.set_ylabel("RMSE mean")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        1.02,
        "Bottom-left is better. Marker numbers map to the detailed legend on the right.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )

    filter_legend = ax.legend(handles=legend_handles, title="Filter Family", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10, title_fontsize=11)
    ax.add_artist(filter_legend)
    ax.legend(handles=point_handles, title="Case Legend", loc="center left", bbox_to_anchor=(1.02, 0.45), fontsize=9, title_fontsize=11, frameon=True)

    scatter_plot = out_dir / f"{csv_path.stem}_tradeoff.png"
    fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    _save_figure(fig, scatter_plot)
    saved.append(scatter_plot)
    return saved


def plot_raw_csv(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        return []

    rmse_groups: dict[str, list[float]] = defaultdict(list)
    runtime_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = f"{row['filter']} | {row['case']}"
        rmse_groups[label].append(_float(row, "rmse_position"))
        runtime_groups[label].append(_float(row, "runtime_sec"))

    labels = list(rmse_groups.keys())
    rmse_data = [rmse_groups[label] for label in labels]
    runtime_data = [runtime_groups[label] for label in labels]
    colors = [FILTER_COLORS.get(label.split(" | ", 1)[0], "tab:gray") for label in labels]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    box = ax.boxplot(rmse_data, tick_labels=labels, patch_artist=True, widths=0.58)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.32)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    for whisker in box["whiskers"]:
        whisker.set_color("gray")
    for cap in box["caps"]:
        cap.set_color("gray")

    ax.set_title("RMSE Distribution by Filter Case")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=22, labelsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.01,
        1.05,
        "Each box shows trial-to-trial RMSE spread for one filter case. Lower boxes mean better accuracy; narrower boxes mean more stable performance.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )

    y_max = max(max(values) for values in rmse_data if values)
    y_min = min(min(values) for values in rmse_data if values)
    y_margin = max((y_max - y_min) * 0.22, 0.04)
    ax.set_ylim(max(0.0, y_min - y_margin * 0.2), y_max + y_margin)
    text_y = y_max + y_margin * 0.15
    for idx, values in enumerate(runtime_data, start=1):
        runtime_mean = sum(values) / len(values)
        ax.text(
            idx,
            text_y,
            f"avg runtime\n{runtime_mean:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.5, label="Box: middle 50% of RMSE trials"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Black line: median RMSE"),
        Line2D([0], [0], color="gray", linewidth=1.2, label="Whiskers: spread outside the box"),
        Line2D([0], [0], color="dimgray", linewidth=0, marker="", label="Text above each case: average runtime"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, title="How to read", title_fontsize=10)

    dist_plot = out_dir / f"{csv_path.stem}_boxplots.png"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(fig, dist_plot)
    saved.append(dist_plot)
    return saved


def plot_per_filter_summary(csv_path: Path, out_dir: Path) -> list[Path]:
    rows = _read_csv(csv_path)
    if not rows:
        return []

    dataset_cases = []
    filter_cases = []
    rmse_map: dict[tuple[str, str], float] = {}
    runtime_map: dict[tuple[str, str], float] = {}

    for row in rows:
        dataset_case = row["dataset_case"]
        filter_case = row["filter_case"]
        if dataset_case not in dataset_cases:
            dataset_cases.append(dataset_case)
        if filter_case not in filter_cases:
            filter_cases.append(filter_case)
        rmse_map[(dataset_case, filter_case)] = _float(row, "rmse_mean")
        runtime_map[(dataset_case, filter_case)] = _float(row, "runtime_mean")

    x = list(range(len(dataset_cases)))
    width = 0.8 / max(len(filter_cases), 1)
    saved: list[Path] = []

    for metric_name, metric_map, ylabel in (
        ("rmse_mean", rmse_map, "RMSE"),
        ("runtime_mean", runtime_map, "Runtime (sec)"),
    ):
        fig, ax = plt.subplots(figsize=(max(10, len(dataset_cases) * 1.8), 5.5))
        for idx, filter_case in enumerate(filter_cases):
            offsets = [base + (idx - (len(filter_cases) - 1) / 2.0) * width for base in x]
            values = [metric_map.get((dataset_case, filter_case), float("nan")) for dataset_case in dataset_cases]
            ax.bar(offsets, values, width=width, label=filter_case, alpha=0.85)

        ax.set_title(f"{csv_path.stem}: {metric_name}")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_cases, rotation=20, ha="right")
        ax.legend(fontsize=8, ncols=2)
        ax.grid(axis="y", alpha=0.2)

        plot_path = out_dir / f"{csv_path.stem}_{metric_name}.png"
        fig.tight_layout()
        _save_figure(fig, plot_path)
        saved.append(plot_path)

    return saved


def _collect_targets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark CSV files into PNG summaries.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Benchmark CSV file or directory to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Defaults to <input>/plots when --input is a directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        input_path.parent / "plots" if input_path.is_file() else input_path / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for csv_path in _collect_targets(input_path):
        name = csv_path.name
        if name == "filter_benchmark_summary.csv":
            saved_paths.extend(plot_summary_csv(csv_path, output_dir))
        elif name == "filter_benchmark_raw.csv":
            saved_paths.extend(plot_raw_csv(csv_path, output_dir))
        elif name.endswith("_per_filter_summary.csv"):
            family_dir = output_dir / _safe_name(csv_path.stem)
            saved_paths.extend(plot_per_filter_summary(csv_path, family_dir))

    if not saved_paths:
        print(f"No supported benchmark CSV files found under: {input_path}")
        return

    print("Generated plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
