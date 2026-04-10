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
    "raw": "tab:gray",
    "ekf": "tab:blue",
    "pf": "tab:orange",
    "ukf": "tab:green",
    "inekf": "tab:red",
}
FILTER_ORDER = {
    "raw": 0,
    "pf": 1,
    "ukf": 2,
    "ekf": 3,
    "inekf": 4,
}
SCENARIO_BG_COLORS = {
    "experiment_2": {
        "gnss_outlier_low": "#eaf4ff",
        "gnss_outlier_medium": "#e8f8f2",
        "gnss_outlier_high": "#fff4e5",
        "gnss_outlier_very_high": "#fdecec",
    },
    "experiment_4": {
        "imu_noise_low": "#eef9ff",
        "imu_noise_high": "#fff1e8",
    },
}
EXPERIMENT_BG_COLORS = {
    "experiment_1": "#e7f0ff",
    "experiment_2": "#fff0e4",
    "experiment_4": "#e9f8ec",
    "unassigned": "#f2f2f2",
}


def _canonical_filter_name(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"imu_integration", "raw", "nofilter", "no_filter"}:
        return "raw"
    return key


def _display_filter_name(name: str) -> str:
    canonical = _canonical_filter_name(name)
    if canonical == "raw":
        return "RAW"
    return canonical.upper()


def _scenario_rank_for_experiment(experiment_name: str, scenario_name: str) -> int:
    exp = str(experiment_name).strip().lower()
    scenario = str(scenario_name).strip().lower()

    if exp == "experiment_2":
        order = {
            "gnss_outlier_low": 0,
            "gnss_outlier_medium": 1,
            "gnss_outlier_high": 2,
            "gnss_outlier_very_high": 3,
        }
        return order.get(scenario, 999)

    if exp == "experiment_4":
        order = {
            "imu_noise_low": 0,
            "imu_noise_high": 1,
        }
        return order.get(scenario, 999)

    if exp == "experiment_1":
        order = {
            "gnss_nominal": 0,
        }
        return order.get(scenario, 999)

    return 999


def _experiment_row_sort_key(experiment_name: str, row: dict[str, str]) -> tuple[int, str, int, str, float]:
    scenario = str(row.get("scenario_case", "")).strip().lower()
    filter_key = _canonical_filter_name(row.get("filter", ""))
    case_name = str(row.get("case", "")).strip().lower()
    rmse = _float(row, "rmse_mean")
    return (
        _scenario_rank_for_experiment(experiment_name, scenario),
        scenario,
        FILTER_ORDER.get(filter_key, 999),
        case_name,
        rmse,
    )


def _experiment_subtitle(experiment_name: str) -> str:
    key = str(experiment_name).strip().lower()
    mapping = {
        "experiment_1": "nominal GNSS baseline",
        "experiment_2": "GNSS outlier sweep",
        "experiment_4": "IMU noise sweep",
    }
    return mapping.get(key, "")


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
    rows_for_bars = [
        row for row in rows if _canonical_filter_name(row.get("filter", "")) != "raw"
    ]
    if not rows_for_bars:
        rows_for_bars = rows

    labels = []
    exp_keys_for_bars: list[str] = []
    for row in rows_for_bars:
        experiment = str(row.get("experiment", "")).strip()
        scenario = str(row.get("scenario_case", "")).strip()
        scope = f"{experiment}/{scenario}" if experiment or scenario else "scenario"
        filter_name = _display_filter_name(row.get("filter", ""))
        case_name = str(row.get("case", "")).strip()
        pose = str(row.get("pose_type", "")).strip()
        mode = str(row.get("mode", "")).strip()
        labels.append(f"{scope} | {filter_name} | {case_name} | {pose}/{mode}")
        exp_keys_for_bars.append((experiment or "unassigned").lower())

    rmse = [_float(row, "rmse_mean") for row in rows_for_bars]
    runtime = [_float(row, "runtime_mean") for row in rows_for_bars]
    y = list(range(len(rows_for_bars)))
    colors = [FILTER_COLORS.get(_canonical_filter_name(row.get("filter", "")), "tab:gray") for row in rows_for_bars]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(14.2, max(6.2, len(rows_for_bars) * 0.92)))
    for idx, exp_key in enumerate(exp_keys_for_bars):
        bg_color = EXPERIMENT_BG_COLORS.get(exp_key)
        if bg_color:
            ax.axhspan(idx - 0.5, idx + 0.5, color=bg_color, alpha=0.85, zorder=0)

    bars = ax.barh(y, rmse, color=colors, alpha=0.88, zorder=2)
    ax.set_title("RMSE Mean by Scenario/Filter Case (RAW excluded)")
    ax.set_xlabel("RMSE mean")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    positive_rmse = [value for value in rmse if value > 0.0]
    dynamic_ratio = (max(positive_rmse) / min(positive_rmse)) if positive_rmse else 1.0
    use_log_scale = dynamic_ratio >= 30.0

    if use_log_scale and positive_rmse:
        x_min = max(min(positive_rmse) * 0.8, 1e-6)
        x_max = max(positive_rmse) * 2.2
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        helper = "Lower RMSE is better. Log scale enabled due to wide RMSE range."
    else:
        x_max = max(rmse) if rmse else 0.0
        ax.set_xlim(0.0, max(x_max * 1.45, 0.02))
        helper = "Lower RMSE is better. Linear scale."

    ax.text(
        0.01,
        1.02,
        helper + " RAW is excluded. Row background color indicates experiment group. Text at bar end shows RMSE and runtime.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )

    linear_offset = max((max(rmse) if rmse else 0.0) * 0.02, 0.004)
    for idx, bar in enumerate(bars):
        if use_log_scale:
            text_x = max(bar.get_width() * 1.06, 1e-6)
        else:
            text_x = bar.get_width() + linear_offset
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"rmse {rmse[idx]:.4f} | t {runtime[idx]:.3f}s",
            ha="left",
            va="center",
            fontsize=8.8,
            color="dimgray",
            clip_on=False,
        )

    summary_plot = out_dir / f"{csv_path.stem}_bars.png"
    fig.tight_layout()
    _save_figure(fig, summary_plot)
    saved.append(summary_plot)

    # Extra figure: grouped by experiment for easier within-experiment comparison.
    experiment_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_for_bars:
        experiment_key = str(row.get("experiment", "")).strip() or "unassigned"
        experiment_groups[experiment_key].append(row)

    grouped_experiments = sorted(experiment_groups.keys())
    if grouped_experiments:
        fig, axes = plt.subplots(
            nrows=len(grouped_experiments),
            ncols=1,
            figsize=(14.8, max(6.0, 3.5 * len(grouped_experiments))),
            squeeze=False,
        )
        axes = axes.flatten()

        for axis_idx, experiment_name in enumerate(grouped_experiments):
            ax_exp = axes[axis_idx]
            exp_rows = sorted(experiment_groups[experiment_name], key=lambda row: _experiment_row_sort_key(experiment_name, row))
            exp_labels = []
            exp_rmse = []
            exp_runtime = []
            exp_colors = []
            exp_scenarios = []

            for row in exp_rows:
                scenario = str(row.get("scenario_case", "")).strip() or "scenario"
                scenario_key = scenario.lower()
                filter_name = _display_filter_name(row.get("filter", ""))
                case_name = str(row.get("case", "")).strip()
                pose = str(row.get("pose_type", "")).strip()
                mode = str(row.get("mode", "")).strip()
                exp_labels.append(f"{scenario} | {filter_name} | {case_name} | {pose}/{mode}")
                exp_rmse.append(_float(row, "rmse_mean"))
                exp_runtime.append(_float(row, "runtime_mean"))
                exp_colors.append(FILTER_COLORS.get(_canonical_filter_name(row.get("filter", "")), "tab:gray"))
                exp_scenarios.append(scenario_key)

            y_exp = list(range(len(exp_rows)))

            exp_bg_map = SCENARIO_BG_COLORS.get(str(experiment_name).strip().lower(), {})
            if exp_bg_map and exp_scenarios:
                segment_start = 0
                while segment_start < len(exp_scenarios):
                    scenario_key = exp_scenarios[segment_start]
                    segment_end = segment_start
                    while segment_end + 1 < len(exp_scenarios) and exp_scenarios[segment_end + 1] == scenario_key:
                        segment_end += 1
                    bg_color = exp_bg_map.get(scenario_key)
                    if bg_color:
                        ax_exp.axhspan(segment_start - 0.5, segment_end + 0.5, color=bg_color, alpha=0.85, zorder=0)
                    segment_start = segment_end + 1

            bars_exp = ax_exp.barh(y_exp, exp_rmse, color=exp_colors, alpha=0.88, zorder=2)
            ax_exp.set_yticks(y_exp)
            ax_exp.set_yticklabels(exp_labels, fontsize=8.5)
            ax_exp.invert_yaxis()
            ax_exp.grid(axis="x", alpha=0.2)
            subtitle = _experiment_subtitle(experiment_name)
            if subtitle:
                ax_exp.set_title(f"{experiment_name} ({subtitle}) | {len(exp_rows)} cases", fontsize=11)
            else:
                ax_exp.set_title(f"{experiment_name} | {len(exp_rows)} cases", fontsize=11)
            ax_exp.set_xlabel("RMSE mean")

            exp_positive = [value for value in exp_rmse if value > 0.0]
            exp_use_log = str(experiment_name).strip().lower() == "experiment_2"

            if exp_use_log and exp_positive:
                exp_x_min = max(min(exp_positive) * 0.8, 1e-6)
                exp_x_max = max(exp_positive) * 2.2
                ax_exp.set_xscale("log")
                ax_exp.set_xlim(exp_x_min, exp_x_max)
            else:
                exp_x_min = 0.0
                exp_x_max = max(exp_rmse, default=0.0) * 1.45
                exp_x_max = max(exp_x_max, 0.02)
                ax_exp.set_xscale("linear")
                ax_exp.set_xlim(exp_x_min, exp_x_max)

            exp_linear_offset = max((max(exp_rmse) if exp_rmse else 0.0) * 0.02, 0.004)
            for idx, bar in enumerate(bars_exp):
                if exp_use_log:
                    txt_x = max(bar.get_width() * 1.06, 1e-6)
                else:
                    txt_x = bar.get_width() + exp_linear_offset
                ax_exp.text(
                    txt_x,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"rmse {exp_rmse[idx]:.4f} | t {exp_runtime[idx]:.3f}s",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color="dimgray",
                    clip_on=False,
                )

        helper_msg = "Grouped by experiment (RAW excluded). experiment_2/4 use scenario background colors. experiment_2: low→medium→high→very_high (log), experiment_4: case-grouped order (linear)."
        fig.text(0.01, 0.995, helper_msg, ha="left", va="top", fontsize=10, color="dimgray")

        grouped_plot = out_dir / f"{csv_path.stem}_bars_by_experiment.png"
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        _save_figure(fig, grouped_plot)
        saved.append(grouped_plot)

    tradeoff_rows = rows_for_bars if rows_for_bars else rows

    fig, ax = plt.subplots(figsize=(19.5, 8.2))
    legend_handles = []
    seen_filters: set[str] = set()
    for row in tradeoff_rows:
        filter_key = _canonical_filter_name(row.get("filter", ""))
        if filter_key in seen_filters:
            continue
        seen_filters.add(filter_key)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=FILTER_COLORS.get(filter_key, "tab:gray"),
                markersize=10,
                label=_display_filter_name(filter_key),
            )
        )

    point_handles = []
    for idx, row in enumerate(tradeoff_rows, start=1):
        color = FILTER_COLORS.get(_canonical_filter_name(row.get("filter", "")), "tab:gray")
        row_rmse = _float(row, "rmse_mean")
        row_runtime = _float(row, "runtime_mean")
        ax.scatter(row_runtime, row_rmse, s=95, color=color, alpha=0.9, edgecolor="black", linewidth=0.4)
        ax.annotate(str(idx), (row_runtime, row_rmse), textcoords="offset points", xytext=(5, 4), fontsize=9, weight="bold")
        experiment = str(row.get("experiment", "")).strip()
        scenario = str(row.get("scenario_case", "")).strip()
        scope = f"{experiment}/{scenario}" if experiment or scenario else "scenario"
        point_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=9,
                label=f"{idx}. {scope} | {_display_filter_name(row.get('filter', ''))} | {row['case']} | rmse={row_rmse:.4f} | t={row_runtime:.3f}s",
            )
        )

    ax.set_title("RMSE Mean vs Runtime (RAW excluded)")
    ax.set_xlabel("Runtime mean (sec)")
    ax.set_ylabel("RMSE mean")
    ax.grid(alpha=0.25)
    fig.text(
        0.06,
        0.985,
        "Bottom-left is better. RAW is excluded here. Marker numbers map to the detailed legend on the right.",
        ha="left",
        va="top",
        fontsize=10,
        color="dimgray",
    )

    filter_legend = ax.legend(handles=legend_handles, title="Filter Family", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10, title_fontsize=11)
    ax.add_artist(filter_legend)
    ax.legend(handles=point_handles, title="Case Legend", loc="center left", bbox_to_anchor=(1.02, 0.45), fontsize=8.5, title_fontsize=11, frameon=True)

    scatter_plot = out_dir / f"{csv_path.stem}_tradeoff.png"
    fig.tight_layout(rect=(0.0, 0.0, 0.66, 0.96))
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
        filter_key = _canonical_filter_name(row.get("filter", ""))
        if filter_key == "raw":
            continue
        label = f"{filter_key} | {row['case']}"
        rmse_groups[label].append(_float(row, "rmse_position"))
        runtime_groups[label].append(_float(row, "runtime_sec"))

    if not rmse_groups:
        return []

    raw_labels = list(rmse_groups.keys())
    family_counts: dict[str, int] = defaultdict(int)
    for label in raw_labels:
        filter_name = label.split(" | ", 1)[0]
        family_counts[_canonical_filter_name(filter_name)] += 1

    def _sort_key(label: str) -> tuple[int, str, str]:
        filter_name, case_name = label.split(" | ", 1)
        canonical = _canonical_filter_name(filter_name)
        return (FILTER_ORDER.get(canonical, 999), canonical, case_name.lower())

    labels = sorted(raw_labels, key=_sort_key)

    display_labels: list[str] = []
    for label in labels:
        filter_name, case_name = label.split(" | ", 1)
        canonical = _canonical_filter_name(filter_name)
        family_name = _display_filter_name(filter_name)
        if family_counts[canonical] > 1:
            display_labels.append(f"{family_name} | {case_name}")
        else:
            display_labels.append(family_name)

    rmse_data = [rmse_groups[label] for label in labels]
    runtime_data = [runtime_groups[label] for label in labels]
    colors = [FILTER_COLORS.get(_canonical_filter_name(label.split(" | ", 1)[0]), "tab:gray") for label in labels]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(14.6, max(7.2, len(labels) * 1.15)))
    box = ax.boxplot(
        rmse_data,
        tick_labels=display_labels,
        patch_artist=True,
        widths=0.58,
        vert=False,
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none", "markeredgecolor": "dimgray", "alpha": 0.9},
    )
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

    ax.set_title("RMSE Distribution by Filter (PF/UKF/EKF/INEKF order, RAW excluded)", y=1.10, pad=12)
    ax.set_xlabel("RMSE")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.01,
        1.03,
        "Circle points are outlier trials outside whisker range. Text at right shows mean RMSE and average runtime.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
    )

    x_max = max(max(values) for values in rmse_data if values)
    x_min = min(min(values) for values in rmse_data if values)
    x_margin = max((x_max - x_min) * 0.22, 0.05)
    ax.set_xlim(max(0.0, x_min - x_margin * 0.2), x_max + x_margin * 2.2)

    text_x = x_max + x_margin * 0.22
    for idx, values in enumerate(rmse_data, start=1):
        rmse_mean = sum(values) / len(values)
        runtime_mean = sum(runtime_data[idx - 1]) / len(runtime_data[idx - 1])
        ax.text(
            text_x,
            idx,
            f"rmse {rmse_mean:.4f} | t {runtime_mean:.3f}s",
            ha="left",
            va="center",
            fontsize=9,
            color="dimgray",
        )

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.5, label="Box: middle 50% of RMSE trials"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Black line: median RMSE"),
        Line2D([0], [0], color="gray", linewidth=1.2, label="Whiskers: spread outside the box"),
        Line2D([0], [0], color="dimgray", marker="o", markersize=5, linewidth=0, markerfacecolor="none", label="Circles: outlier trials"),
        Line2D([0], [0], color="dimgray", linewidth=0, marker="", label="Right text: mean RMSE and avg runtime"),
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, title="How to read", title_fontsize=10, frameon=True)

    dist_plot = out_dir / f"{csv_path.stem}_boxplots.png"
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.95))
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
