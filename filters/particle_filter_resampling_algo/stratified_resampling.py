from __future__ import annotations

import numpy as np


# Stratified resampling:
# [0,1] 구간을 여러 층으로 나누고 각 구간에서 하나씩 랜덤하게 뽑아 multinomial보다 분산을 줄이는 방식이다.
# 장점: systematic과 비슷하게 low-variance 특성이 있고 particle을 더 고르게 뽑기 쉽다.
# 단점: 구현은 간단하지만 systematic보다 아주 약간 계산량이 늘고, 실무 표준으로는 systematic이 더 자주 선택된다.
def stratified_resample(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return particle indices sampled with stratified resampling."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    n = weights.size
    if n == 0:
        return np.zeros(0, dtype=int)

    total = np.sum(weights)
    if (not np.isfinite(total)) or total <= 0.0:
        weights = np.full(n, 1.0 / n, dtype=float)
    else:
        weights = weights / total

    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0

    generator = np.random.default_rng() if rng is None else rng
    positions = (np.arange(n) + generator.random(n)) / n
    return np.searchsorted(cumulative, positions, side="left")
