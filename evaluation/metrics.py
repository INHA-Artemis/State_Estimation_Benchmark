# [협업 주석]
# Goal: state estimation 성능 metric 계산 모듈을 제공한다.
# What it does: RMSE/ATE/RPE 등 benchmark metric 함수들을 추후 정의할 예정이다.
"""Metric computation utilities for estimator benchmarks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_array(name: str, value: np.ndarray) -> np.ndarray:
    """
    Goal:
        metric 계산 전에 입력 배열 형식을 2D numpy array로 검증한다.
    Input:
        name은 오류 메시지용 식별자이고, value는 검증할 array-like 입력이다.
    Output:
        dtype=float의 2D numpy array를 반환한다.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D array [T, D], got shape={arr.shape}")
    return arr


def _latency_stats(latency_s: np.ndarray) -> dict[str, float]:
    """
    Goal:
        latency sequence를 summary statistics로 변환한다.
    Input:
        latency_s는 seconds 단위 latency numpy array이다.
    Output:
        count, mean_ms, percentile 등을 포함한 dict를 반환한다.
    """
    if latency_s.size == 0:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "total_ms": 0.0,
        }
    values_ms = latency_s * 1_000.0
    return {
        "count": int(values_ms.size),
        "mean_ms": float(np.mean(values_ms)),
        "std_ms": float(np.std(values_ms)),
        "p50_ms": float(np.percentile(values_ms, 50.0)),
        "p95_ms": float(np.percentile(values_ms, 95.0)),
        "max_ms": float(np.max(values_ms)),
        "total_ms": float(np.sum(values_ms)),
    }


def compute_metrics(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    timestamps: np.ndarray | None = None,
    position_indices: Sequence[int] = (0, 1),
    yaw_index: int | None = None,
    step_latency_s: np.ndarray | None = None,
    predict_latency_s: np.ndarray | None = None,
    update_latency_s: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Goal:
        estimation 결과와 ground truth를 비교해 핵심 benchmark metric을 계산한다.
    Input:
        estimates와 ground_truth는 [T, D] 배열이고, timestamps/position_indices/yaw_index는 metric 계산 옵션이다.
        step_latency_s, predict_latency_s, update_latency_s는 latency 통계 계산에 사용된다.
    Output:
        RMSE, position error, yaw_rmse, latency summary가 담긴 JSON-serializable dict를 반환한다.
    """
    est = _to_array("estimates", estimates)
    gt = _to_array("ground_truth", ground_truth)
    if est.shape != gt.shape:
        raise ValueError(f"estimates and ground_truth shape mismatch: {est.shape} vs {gt.shape}")
    if est.shape[0] == 0:
        raise ValueError("No samples provided for metric computation.")

    error = est - gt
    state_rmse_axis = np.sqrt(np.mean(np.square(error), axis=0))
    state_rmse = float(np.sqrt(np.mean(np.square(error))))

    pos_idx = tuple(int(i) for i in position_indices)
    pos_error = error[:, pos_idx]
    pos_norm = np.linalg.norm(pos_error, axis=1)
    pos_rmse = float(np.sqrt(np.mean(np.square(pos_norm))))
    final_pos_error = float(pos_norm[-1])
    max_pos_error = float(np.max(pos_norm))

    yaw_rmse = None
    if yaw_index is not None and yaw_index < error.shape[1]:
        yaw_err = error[:, yaw_index]
        yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi
        yaw_rmse = float(np.sqrt(np.mean(np.square(yaw_err))))

    duration_s = None
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float)
        if ts.ndim != 1:
            raise ValueError("timestamps must be 1D array [T].")
        if ts.size != est.shape[0]:
            raise ValueError("timestamps length must equal sample count.")
        duration_s = float(max(ts[-1] - ts[0], 0.0))

    step_latency = np.asarray(step_latency_s if step_latency_s is not None else [], dtype=float)
    predict_latency = np.asarray(predict_latency_s if predict_latency_s is not None else [], dtype=float)
    update_latency = np.asarray(update_latency_s if update_latency_s is not None else [], dtype=float)
    latency = {
        "step": _latency_stats(step_latency),
        "predict": _latency_stats(predict_latency),
        "update": _latency_stats(update_latency),
    }
    if duration_s is not None and duration_s > 0.0:
        latency["throughput_hz"] = float(est.shape[0] / duration_s)
    else:
        latency["throughput_hz"] = 0.0

    return {
        "num_samples": int(est.shape[0]),
        "state_dim": int(est.shape[1]),
        "state_rmse": state_rmse,
        "state_rmse_axis": [float(v) for v in state_rmse_axis],
        "position_rmse": pos_rmse,  # Equivalent to 2D/3D ATE RMSE over selected position indices.
        "final_position_error": final_pos_error,
        "max_position_error": max_pos_error,
        "yaw_rmse": yaw_rmse,
        "latency": latency,
    }


def compute_latency_only_metrics(
    step_latency_s: np.ndarray | None = None,
    predict_latency_s: np.ndarray | None = None,
    update_latency_s: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Goal:
        ground truth가 없을 때 latency 관련 metric만 계산한다.
    Input:
        step_latency_s, predict_latency_s, update_latency_s는 단계별 latency 배열이다.
    Output:
        latency summary와 sample 수를 담은 dict를 반환한다.
    """
    step_latency = np.asarray(step_latency_s if step_latency_s is not None else [], dtype=float)
    predict_latency = np.asarray(predict_latency_s if predict_latency_s is not None else [], dtype=float)
    update_latency = np.asarray(update_latency_s if update_latency_s is not None else [], dtype=float)
    return {
        "num_samples": int(step_latency.size),
        "latency": {
            "step": _latency_stats(step_latency),
            "predict": _latency_stats(predict_latency),
            "update": _latency_stats(update_latency),
            "throughput_hz": 0.0,
        },
    }
