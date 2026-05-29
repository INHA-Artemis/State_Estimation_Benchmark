from __future__ import annotations

import numpy as np


def resample_indices(method: str, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    method = str(method).lower()
    if method == "systematic":
        return systematic_resample(weights, rng)
    if method == "stratified":
        return stratified_resample(weights, rng)
    if method == "residual":
        return residual_resample(weights, rng)
    if method == "multinomial":
        return multinomial_resample(weights, rng)
    raise ValueError(f"Unknown resampling_method: {method}")


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return _positions_to_indices(weights, positions)


def stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random(n) + np.arange(n)) / n
    return _positions_to_indices(weights, positions)


def multinomial_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    return np.searchsorted(cumulative_sum, rng.random(len(weights)))


def residual_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    indexes = np.zeros(n, dtype=int)
    copies = np.floor(n * weights).astype(int)
    k = 0
    for idx, count in enumerate(copies):
        indexes[k : k + count] = idx
        k += count
    residual = weights - copies
    if np.sum(residual) > 0.0 and k < n:
        residual = residual / np.sum(residual)
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0
        indexes[k:] = np.searchsorted(cumulative_sum, rng.random(n - k))
    return indexes


def _positions_to_indices(weights: np.ndarray, positions: np.ndarray) -> np.ndarray:
    indexes = np.zeros(len(weights), dtype=int)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    i = 0
    j = 0
    while i < len(weights):
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
