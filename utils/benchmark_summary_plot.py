from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILES = {
    "filterpy": "filterpy_kaist_vio_results.csv",
    "stonesoup": "stonesoup_kaist_vio_results.csv",
    "invariant_ekf": "invariant_ekf_kaist_vio_results.csv",
    "drift": "drift_kaist_vio_results.csv",
}

DEFAULT_PF_RMSE_SUMMARY = PROJECT_ROOT / "outputs" / "benchmarks" / "stonesoup_pf_rmse_repeats" / "pf_rmse_summary.csv"
PF_ERROR_COLUMNS = ("rmse_std", "rmse_stderr")
PF_RUNTIME_ERROR_COLUMNS = ("runtime_std", "runtime_stderr")

METRICS = [
    ("rmse_position", "Position RMSE [m]"),
    ("runtime_sec", "Runtime [s]"),
]

REPORTS = [
    {
        "name": "kaist_vio_filterpy_stonesoup",
        "title": "KAIST VIO EKF/UKF/PF: Ours vs FilterPy and Stone Soup",
        "dataset_family": "kaist_vio",
        "runs": ["square", "infinite", "circle", "rotation"],
        "benchmarks": ["filterpy", "stonesoup"],
    },
    {
        "name": "kaist_vio_inekf",
        "title": "KAIST VIO InEKF: Ours vs invariant-ekf and DRIFT",
        "dataset_family": "kaist_vio",
        "runs": ["square", "infinite", "circle", "rotation"],
        "benchmarks": ["invariant_ekf", "drift"],
    },
    {
        "name": "m2dgr_filterpy_stonesoup",
        "title": "M2DGR EKF/UKF/PF: Ours vs FilterPy and Stone Soup",
        "dataset_family": "m2dgr",
        "runs": ["street02"],
        "benchmarks": ["filterpy", "stonesoup"],
    },
    {
        "name": "m2dgr_inekf",
        "title": "M2DGR InEKF: Ours vs invariant-ekf and DRIFT",
        "dataset_family": "m2dgr",
        "runs": ["street02"],
        "benchmarks": ["invariant_ekf", "drift"],
    },
]

IMPLEMENTATION_LABELS = {
    "Our EKF": "Ours EKF",
    "Our UKF": "Ours UKF",
    "Our PF": "Ours PF",
    "Our InEKF": "Ours InEKF",
    "Our InEKF 15D": "Ours InEKF 15D",
    "FilterPy": "FilterPy",
    "Stone Soup": "Stone Soup",
    "invariant-ekf": "invariant-ekf",
    "drift": "DRIFT",
}

ALGORITHM_ORDER = {
    "ekf": 0,
    "ukf": 1,
    "pf": 2,
    "inekf": 3,
}

IMPLEMENTATION_ORDER = {
    "ours": 0,
    "filterpy": 1,
    "stonesoup": 2,
    "invariant_ekf": 3,
    "drift": 4,
}

BENCHMARK_ORDER = {
    "filterpy": 0,
    "stonesoup": 1,
    "invariant_ekf": 2,
    "drift": 3,
}


@dataclass
class MissingResult:
    benchmark: str
    dataset_family: str
    run: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw report-style benchmark summary bar figures.")
    parser.add_argument("--outputs-root", default=str(PROJECT_ROOT / "outputs" / "benchmarks"))
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/benchmarks/summary/report.")
    parser.add_argument("--linear-scale", action="store_true", help="Use linear y axes instead of log-scale axes.")
    parser.add_argument("--single-report", choices=[str(report["name"]) for report in REPORTS], default="")
    parser.add_argument(
        "--pf-rmse-summary",
        default=str(DEFAULT_PF_RMSE_SUMMARY),
        help="Optional CSV from benchmarks/stonesoup_pf_rmse_repeats.py used to add PF RMSE error bars.",
    )
    parser.add_argument(
        "--pf-error-column",
        choices=PF_ERROR_COLUMNS,
        default="rmse_std",
        help="Error-bar column to use from the PF repeat summary CSV.",
    )
    parser.add_argument(
        "--pf-runtime-error-column",
        choices=PF_RUNTIME_ERROR_COLUMNS,
        default="runtime_std",
        help="Runtime error-bar column to use from the PF repeat summary CSV.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir) if args.output_dir else outputs_root / "summary" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    pf_repeat_summary = load_pf_repeat_summary(
        Path(args.pf_rmse_summary),
        args.pf_error_column,
        args.pf_runtime_error_column,
    )

    selected_reports = [report for report in REPORTS if not args.single_report or report["name"] == args.single_report]
    all_missing: list[MissingResult] = []
    for report in selected_reports:
        rows, missing = collect_report_rows(outputs_root, report)
        apply_pf_repeat_summary(rows, pf_repeat_summary)
        all_missing.extend(missing)
        summary_csv = output_dir / f"{report['name']}_summary.csv"
        figure_path = output_dir / f"{report['name']}.png"
        write_summary_csv(summary_csv, rows)
        draw_report_figure(figure_path, rows, report, log_scale=not args.linear_scale)
        print(f"[BenchmarkSummary] CSV   : {summary_csv}")
        print(f"[BenchmarkSummary] Figure: {figure_path}")

    for item in all_missing:
        print(f"[BenchmarkSummary] Missing {item.benchmark} {item.dataset_family}/{item.run}: {item.path}")


def collect_report_rows(outputs_root: Path, report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[MissingResult]]:
    rows: list[dict[str, Any]] = []
    missing: list[MissingResult] = []
    dataset_family = str(report["dataset_family"])
    for run in report["runs"]:
        for benchmark in report["benchmarks"]:
            path = outputs_root / benchmark / dataset_family / str(run) / RESULT_FILES[benchmark]
            if not path.exists():
                missing.append(MissingResult(benchmark, dataset_family, str(run), path))
                continue
            for row in _read_csv_rows(path):
                if row.get("status") != "ok":
                    continue
                implementation = row.get("implementation", "")
                algorithm = row.get("algorithm", "")
                rows.append(
                    {
                        "benchmark": benchmark,
                        "dataset_family": dataset_family,
                        "run": str(run),
                        "family": row.get("family", ""),
                        "implementation": implementation,
                        "algorithm": algorithm,
                        "label": _label(benchmark, row.get("family", ""), implementation, algorithm),
                        "rmse_position": _float_or_nan(row.get("rmse_position")),
                        "error_variance": _float_or_nan(row.get("error_variance")),
                        "runtime_sec": _float_or_nan(row.get("runtime_sec")),
                        "steps": row.get("steps", ""),
                    }
                )
    return _sort_rows(_deduplicate_rows(rows, report), report), missing


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "dataset_family",
        "run",
        "family",
        "implementation",
        "algorithm",
        "label",
        "rmse_position",
        "rmse_error",
        "rmse_repeat_n",
        "rmse_error_source",
        "error_variance",
        "runtime_sec",
        "runtime_error",
        "runtime_repeat_n",
        "runtime_error_source",
        "steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def draw_report_figure(path: Path, rows: list[dict[str, Any]], report: dict[str, Any], log_scale: bool = True) -> None:
    runs = [str(run) for run in report["runs"]]
    if report["dataset_family"] == "kaist_vio":
        plot_runs = runs
        fig, axes = plt.subplots(len(METRICS), len(plot_runs), figsize=(24.0, 8.6), squeeze=False)
        for row_idx, (field, ylabel) in enumerate(METRICS):
            for col_idx, run in enumerate(plot_runs):
                ax = axes[row_idx][col_idx]
                run_rows = _rows_for_plot_run(rows, run, runs)
                _draw_metric_axis(ax, run_rows, run, field, ylabel, log_scale)
    else:
        nrows = len(runs)
        fig, axes = plt.subplots(nrows, len(METRICS), figsize=(12.5, 4.8 * nrows), squeeze=False)
        for row_idx, run in enumerate(runs):
            run_rows = [row for row in rows if row["run"] == run]
            for col_idx, (field, ylabel) in enumerate(METRICS):
                ax = axes[row_idx][col_idx]
                _draw_metric_axis(ax, run_rows, run, field, ylabel, log_scale)

    scale = "log scale" if log_scale else "linear scale"
    fig.suptitle(f"{report['title']} ({scale})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_metric_axis(
    ax: Any,
    run_rows: list[dict[str, Any]],
    run: str,
    field: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    labels = [str(row["label"]) for row in run_rows]
    x = np.arange(len(run_rows), dtype=float)
    colors = [_color_for(row) for row in run_rows]
    values = np.asarray([row[field] for row in run_rows], dtype=float)
    yerr = _metric_yerr(run_rows, field)
    bars = ax.bar(
        x,
        values,
        yerr=yerr,
        error_kw={"elinewidth": 1.2, "ecolor": "#222222", "capsize": 4, "capthick": 1.2},
        color=colors,
        alpha=0.88,
    )
    ax.set_title(f"{run} - {ylabel.split(' [')[0]}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=26, ha="right")
    ax.grid(axis="y", alpha=0.25)

    if log_scale:
        _set_log_axis(ax, values, yerr)
    else:
        _set_linear_axis(ax, values, yerr)
    annotation_tops = _annotation_tops(values, yerr)
    _annotate_bars(ax, bars, values, annotation_tops)

    if not run_rows:
        ax.text(0.5, 0.5, "No successful results", transform=ax.transAxes, ha="center", va="center")


def _rows_for_plot_run(rows: list[dict[str, Any]], run: str, source_runs: list[str]) -> list[dict[str, Any]]:
    if run != "mean":
        return [row for row in rows if row["run"] == run]

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["run"] not in source_runs:
            continue
        key = (
            str(row.get("benchmark", "")),
            str(row.get("family", "")),
            str(row.get("implementation", "")),
            str(row.get("algorithm", "")),
            str(row.get("label", "")),
        )
        grouped.setdefault(key, []).append(row)

    mean_rows: list[dict[str, Any]] = []
    for grouped_rows in grouped.values():
        base = dict(grouped_rows[0])
        base["run"] = "mean"
        for field in ("rmse_position", "error_variance", "runtime_sec"):
            values = np.asarray([row[field] for row in grouped_rows], dtype=float)
            finite = values[np.isfinite(values)]
            base[field] = float(np.mean(finite)) if finite.size else float("nan")
        errors = np.asarray([row.get("rmse_error", float("nan")) for row in grouped_rows], dtype=float)
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size:
            base["rmse_error"] = float(np.mean(finite_errors))
            base["rmse_repeat_n"] = grouped_rows[0].get("rmse_repeat_n", "")
            base["rmse_error_source"] = grouped_rows[0].get("rmse_error_source", "")
        runtime_errors = np.asarray([row.get("runtime_error", float("nan")) for row in grouped_rows], dtype=float)
        finite_runtime_errors = runtime_errors[np.isfinite(runtime_errors)]
        if finite_runtime_errors.size:
            base["runtime_error"] = float(np.mean(finite_runtime_errors))
            base["runtime_repeat_n"] = grouped_rows[0].get("runtime_repeat_n", "")
            base["runtime_error_source"] = grouped_rows[0].get("runtime_error_source", "")
        mean_rows.append(base)
    return _sort_rows(mean_rows, {"runs": ["mean"]})


def load_pf_repeat_summary(
    path: Path,
    error_column: str,
    runtime_error_column: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return _load_pf_repeat_runtime_from_raw(path.with_name("pf_rmse_raw.csv"), {})

    summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _read_csv_rows(path):
        if str(row.get("algorithm", "")).lower() != "pf":
            continue
        dataset_family = str(row.get("dataset_family", "")).strip()
        dataset = _normalize_pf_repeat_dataset(str(row.get("dataset", "")).strip())
        implementation = str(row.get("implementation", "")).strip()
        if not dataset_family or not dataset or implementation not in {"Ours PF", "Stone Soup"}:
            continue
        summary[(dataset_family, dataset, implementation)] = {
            "rmse_mean": _float_or_nan(row.get("rmse_mean")),
            "rmse_error": _float_or_nan(row.get(error_column)),
            "rmse_repeat_n": row.get("n", ""),
            "rmse_error_source": error_column,
            "runtime_mean": _float_or_nan(row.get("runtime_mean")),
            "runtime_error": _float_or_nan(row.get(runtime_error_column)),
            "runtime_repeat_n": row.get("n", ""),
            "runtime_error_source": runtime_error_column,
        }
    return _load_pf_repeat_runtime_from_raw(path.with_name("pf_rmse_raw.csv"), summary)


def apply_pf_repeat_summary(
    rows: list[dict[str, Any]],
    pf_repeat_summary: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    if not pf_repeat_summary:
        return

    for row in rows:
        if str(row.get("algorithm", "")).lower() != "pf":
            continue
        implementation = _pf_repeat_implementation(row)
        if implementation is None:
            continue
        dataset_family = str(row.get("dataset_family", ""))
        dataset = _normalize_pf_repeat_dataset(str(row.get("run", "")))
        repeat = pf_repeat_summary.get((dataset_family, dataset, implementation))
        if repeat is None:
            continue

        rmse_mean = repeat["rmse_mean"]
        if np.isfinite(rmse_mean):
            row["rmse_position"] = rmse_mean
        row["rmse_error"] = repeat["rmse_error"]
        row["rmse_repeat_n"] = repeat["rmse_repeat_n"]
        row["rmse_error_source"] = repeat["rmse_error_source"]
        runtime_mean = repeat.get("runtime_mean", float("nan"))
        if np.isfinite(runtime_mean):
            row["runtime_sec"] = runtime_mean
        row["runtime_error"] = repeat.get("runtime_error", float("nan"))
        row["runtime_repeat_n"] = repeat.get("runtime_repeat_n", "")
        row["runtime_error_source"] = repeat.get("runtime_error_source", "")


def _load_pf_repeat_runtime_from_raw(
    path: Path,
    summary: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return summary

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in _read_csv_rows(path):
        if str(row.get("algorithm", "")).lower() != "pf" or row.get("status") != "ok":
            continue
        dataset_family = str(row.get("dataset_family", "")).strip()
        dataset = _normalize_pf_repeat_dataset(str(row.get("dataset", "")).strip())
        implementation = str(row.get("implementation", "")).strip()
        if not dataset_family or not dataset or implementation not in {"Ours PF", "Stone Soup"}:
            continue
        runtime = _float_or_nan(row.get("runtime_sec"))
        if np.isfinite(runtime):
            grouped.setdefault((dataset_family, dataset, implementation), []).append(runtime)

    for key, values_list in grouped.items():
        values = np.asarray(values_list, dtype=float)
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        target = summary.setdefault(
            key,
            {
                "rmse_mean": float("nan"),
                "rmse_error": float("nan"),
                "rmse_repeat_n": "",
                "rmse_error_source": "",
            },
        )
        if not np.isfinite(_float_or_nan(target.get("runtime_mean"))):
            target["runtime_mean"] = float(np.mean(values))
        if not np.isfinite(_float_or_nan(target.get("runtime_error"))):
            target["runtime_error"] = std
            target["runtime_error_source"] = "runtime_std_from_raw"
        target["runtime_repeat_n"] = target.get("runtime_repeat_n") or str(values.size)
    return summary


def _pf_repeat_implementation(row: dict[str, Any]) -> str | None:
    family = str(row.get("family", ""))
    implementation = str(row.get("implementation", ""))
    benchmark = str(row.get("benchmark", ""))
    if family == "ours" and implementation == "Our PF":
        return "Ours PF"
    if benchmark == "stonesoup" and implementation == "Stone Soup":
        return "Stone Soup"
    return None


def _normalize_pf_repeat_dataset(value: str) -> str:
    normalized = value.strip().lower().replace("_", "")
    if normalized in {"street02", "m2dgr", "m2dgrstreet02"}:
        return "m2dgr"
    return normalized


def _metric_yerr(run_rows: list[dict[str, Any]], field: str) -> np.ndarray | None:
    return None


def _annotation_tops(values: np.ndarray, yerr: np.ndarray | None) -> np.ndarray:
    if yerr is None:
        return values
    return values + np.where(np.isfinite(yerr), yerr, 0.0)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _deduplicate_rows(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    benchmarks = set(report.get("benchmarks", []))
    if benchmarks != {"filterpy", "stonesoup"}:
        return rows

    seen_ours: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("run", "")), str(row.get("family", "")), str(row.get("algorithm", "")))
        if row.get("family") == "ours":
            if key in seen_ours:
                continue
            seen_ours.add(key)
        deduped.append(row)
    return deduped


def _sort_rows(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    run_order = {str(run): idx for idx, run in enumerate(report["runs"])}
    return sorted(rows, key=lambda row: _row_sort_key(row, run_order))


def _row_sort_key(row: dict[str, Any], run_order: dict[str, int]) -> tuple[int, int, int, int, int, str]:
    algorithm = str(row.get("algorithm", "")).lower()
    benchmark = str(row.get("benchmark", ""))
    family = str(row.get("family", ""))
    implementation = str(row.get("implementation", ""))
    implementation_key = benchmark if family != "ours" else "ours"
    return (
        run_order.get(str(row.get("run", "")), 999),
        ALGORITHM_ORDER.get(algorithm, 99),
        BENCHMARK_ORDER.get(benchmark, 99),
        IMPLEMENTATION_ORDER.get(implementation_key, 99),
        _ours_variant_order(implementation),
        str(row.get("label", "")),
    )


def _ours_variant_order(implementation: str) -> int:
    if implementation == "Our InEKF":
        return 0
    if implementation == "Our InEKF 15D":
        return 1
    return 0


def _label(benchmark: str, family: str, implementation: str, algorithm: str) -> str:
    implementation_label = IMPLEMENTATION_LABELS.get(implementation, implementation)
    if family == "ours":
        if benchmark == "invariant_ekf":
            return f"{implementation_label} (inv input)"
        if benchmark == "drift":
            return f"{implementation_label} (drift input)"
        return implementation_label
    return f"{implementation_label} {algorithm.upper()}"


def _float_or_nan(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _color_for(row: dict[str, Any]) -> str:
    if row.get("family") == "ours":
        return "#4C78A8"
    benchmark = row.get("benchmark")
    if benchmark == "filterpy":
        return "#72B7B2"
    if benchmark == "stonesoup":
        return "#F58518"
    if benchmark == "invariant_ekf":
        return "#54A24B"
    if benchmark == "drift":
        return "#B279A2"
    return "#777777"


def _annotate_bars(ax: Any, bars: Any, values: np.ndarray, annotation_tops: np.ndarray | None = None) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    is_log = ax.get_yscale() == "log"
    tops = values if annotation_tops is None else annotation_tops
    finite_tops = tops[np.isfinite(tops)]
    offset = max(float(np.nanmax(finite_tops if finite_tops.size else finite)) * 0.015, 1e-12)
    for bar, value, top in zip(bars, values, tops):
        if not np.isfinite(value):
            label = "nan"
            height = 0.0
        else:
            label = f"{value:.3g}"
            height = float(value)
        annotation_top = float(top) if np.isfinite(top) else height
        y = annotation_top * 1.08 if is_log and annotation_top > 0.0 else annotation_top + offset
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def _set_log_axis(ax: Any, values: np.ndarray, yerr: np.ndarray | None = None) -> None:
    tops = _annotation_tops(values, yerr)
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        return
    ymin = float(np.nanmin(finite_positive))
    finite_tops = tops[np.isfinite(tops) & (tops > 0.0)]
    ymax = float(np.nanmax(finite_tops if finite_tops.size else finite_positive))
    ax.set_yscale("log")
    ax.set_ylim(max(ymin * 0.45, 1e-12), ymax * 3.2)
    ax.grid(axis="y", which="both", alpha=0.22)


def _set_linear_axis(ax: Any, values: np.ndarray, yerr: np.ndarray | None = None) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    tops = _annotation_tops(values, yerr)
    finite_tops = tops[np.isfinite(tops)]
    ymin = min(0.0, float(np.nanmin(finite)))
    ymax = float(np.nanmax(finite_tops if finite_tops.size else finite))
    span = ymax - ymin
    headroom = max(span * 0.22, ymax * 0.18, 1e-9)
    ax.set_ylim(ymin, ymax + headroom)


if __name__ == "__main__":
    main()
