"""Small math helpers shared across modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def weighted_mean_cov(
    samples: np.ndarray,
    weights: np.ndarray,
    angle_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean/covariance with optional circular mean for angles."""
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
