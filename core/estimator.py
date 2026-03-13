# [협업 주석]
# Goal: filter-independent estimator 실행 래퍼를 제공한다.
# What it does: filter lifecycle(init/predict/update/reset) orchestration 공통 로직을 담을 예정이다.
"""Filter execution engine for dataset sequences."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from datasets.dataset_base import DatasetSequence
from filters.base_filter import BaseFilter

RunMode = Literal["imu_only", "gps_only", "fused"]


@dataclass
class EstimationResult:
    """Container for one filter execution over one sequence."""

    timestamps: np.ndarray
    estimates: np.ndarray
    ground_truth: np.ndarray
    gps_measurements: np.ndarray
    step_latency_s: np.ndarray
    predict_latency_s: np.ndarray
    update_latency_s: np.ndarray


def run_filter_sequence(
    filter_obj: BaseFilter,
    sequence: DatasetSequence,
    mode: RunMode,
    state_dim: int,
) -> EstimationResult:
    """Execute one filter on one dataset sequence and collect raw traces."""
    if mode not in {"imu_only", "gps_only", "fused"}:
        raise ValueError(f"Unsupported mode: {mode}")

    estimates: list[np.ndarray] = []
    gt_states: list[np.ndarray] = []
    gps_measurements: list[np.ndarray] = []
    times: list[float] = []
    step_latency: list[float] = []
    predict_latency: list[float] = []
    update_latency: list[float] = []

    for step in sequence:
        t0 = perf_counter()
        pred_dt = 0.0
        upd_dt = 0.0

        if mode in {"imu_only", "fused"}:
            if step.imu is None:
                raise ValueError("IMU data required for selected mode but not present in dataset step.")
            pred_t0 = perf_counter()
            filter_obj.predict(step.imu, step.dt)
            pred_dt = perf_counter() - pred_t0
        else:
            pred_t0 = perf_counter()
            filter_obj.predict(None, step.dt)
            pred_dt = perf_counter() - pred_t0

        if mode in {"gps_only", "fused"} and step.gps is not None:
            upd_t0 = perf_counter()
            filter_obj.update(step.gps)
            upd_dt = perf_counter() - upd_t0

        est = filter_obj.get_state()
        estimates.append(np.asarray(est, dtype=float))
        times.append(float(step.t))
        step_latency.append(perf_counter() - t0)
        predict_latency.append(pred_dt)
        update_latency.append(upd_dt)

        if step.gt_state is not None:
            gt = np.asarray(step.gt_state[:state_dim], dtype=float)
        else:
            gt = np.full((state_dim,), np.nan, dtype=float)
        gt_states.append(gt)

        if step.gps is not None:
            gps_measurements.append(np.asarray(step.gps, dtype=float).reshape(-1))
        else:
            gps_measurements.append(np.full((2,), np.nan, dtype=float))

    return EstimationResult(
        timestamps=np.asarray(times, dtype=float),
        estimates=np.asarray(estimates, dtype=float),
        ground_truth=np.asarray(gt_states, dtype=float),
        gps_measurements=np.asarray(gps_measurements, dtype=float),
        step_latency_s=np.asarray(step_latency, dtype=float),
        predict_latency_s=np.asarray(predict_latency, dtype=float),
        update_latency_s=np.asarray(update_latency, dtype=float),
    )
