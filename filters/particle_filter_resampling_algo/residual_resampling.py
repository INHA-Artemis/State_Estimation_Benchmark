from __future__ import annotations

import numpy as np


# Residual resampling:
# 큰 weight를 가진 particle은 먼저 deterministic하게 복제하고, 남은 개수만 확률적으로 추출해 분산을 줄이는 방식이다.
# 장점: 큰 weight particle을 안정적으로 보존해 multinomial보다 variance를 줄이기 쉽다.
# 단점: 구현이 조금 더 복잡하고, deterministic 복제가 많으면 sample impoverishment가 생길 수 있다.
def residual_resample(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return particle indices sampled with residual resampling."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    n = weights.size
    if n == 0:
        return np.zeros(0, dtype=int)

    total = np.sum(weights)
    if (not np.isfinite(total)) or total <= 0.0:
        weights = np.full(n, 1.0 / n, dtype=float)
    else:
        weights = weights / total

    generator = np.random.default_rng() if rng is None else rng

    copies = np.floor(n * weights).astype(int)
    indices = np.repeat(np.arange(n, dtype=int), copies)

    residual_count = n - indices.size
    if residual_count <= 0:
        return indices[:n]

    residual = n * weights - copies
    residual_sum = np.sum(residual)
    if residual_sum > 0.0:
        residual = residual / residual_sum
        extra = generator.choice(n, size=residual_count, p=residual)
    else:
        extra = generator.choice(n, size=residual_count)

    return np.concatenate([indices, extra])
