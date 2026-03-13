# [협업 주석]
# Goal: filter/model에서 공통으로 사용하는 수학 helper를 분리한다.
# What it does: angle wrapping과 weighted mean/covariance 계산(angular state 보정 포함)을 제공한다.
"""Small math helpers shared across modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """
    Goal:
        angle 값을 [-pi, pi) 범위로 정규화한다.
    Input:
        angle은 scalar 또는 numpy array 형태의 angle 값이다.
    Output:
        wrap된 angle 값을 같은 의미의 numpy scalar/array로 반환한다.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def weighted_mean_cov(
    samples: np.ndarray,
    weights: np.ndarray,
    angle_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Goal:
        weighted sample 집합의 mean과 covariance를 계산한다.
    Input:
        samples는 [N, D] sample array이고, weights는 각 sample weight이다.
        angle_indices가 주어지면 해당 차원은 circular mean과 wrapped residual로 처리한다.
    Output:
        weighted mean vector와 covariance matrix tuple을 반환한다.
    """
    angle_set = set(angle_indices or [])
    mean = np.average(samples, axis=0, weights=weights)

    for idx in angle_set:
        s = np.sin(samples[:, idx])
        c = np.cos(samples[:, idx])
        mean[idx] = np.arctan2(np.average(s, weights=weights), np.average(c, weights=weights))

    centered = samples - mean
    for idx in angle_set:
        centered[:, idx] = wrap_angle(centered[:, idx])

    cov = centered.T @ (centered * weights[:, None])
    return mean, cov
