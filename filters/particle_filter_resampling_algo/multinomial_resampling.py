from __future__ import annotations

import numpy as np


# Multinomial resampling:
# 각 particle을 weight 분포에 따라 독립적으로 다시 추출하는 가장 기본적인 resampling 방식이다.
# 장점: 구현이 가장 단순하고 개념적으로 이해하기 쉽다.
# 단점: resampling variance가 큰 편이라 particle diversity가 쉽게 흔들릴 수 있다.
def multinomial_resample(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return particle indices sampled with multinomial resampling."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        return np.zeros(0, dtype=int)

    total = np.sum(weights)
    if (not np.isfinite(total)) or total <= 0.0:
        weights = np.full(weights.size, 1.0 / weights.size, dtype=float)
    else:
        weights = weights / total

    generator = np.random.default_rng() if rng is None else rng
    return generator.choice(weights.size, size=weights.size, p=weights)
