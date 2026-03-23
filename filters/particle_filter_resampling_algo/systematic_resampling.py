from __future__ import annotations

import numpy as np


# Systematic resampling:
# 하나의 랜덤 offset과 균일 간격 샘플을 사용해 전체 누적분포를 훑으며 particle을 다시 뽑는 효율적인 방식이다.
# 장점: 계산이 빠르고 variance가 비교적 낮아 실무와 연구 코드에서 매우 널리 쓰인다.
# 단점: 특정 경우에는 stratified/residual보다 다양성 보존이 아주 약간 불리할 수 있다.
# 참고: 네 가지 중에서 가장 널리 쓰이는 resampling 방법 중 하나이며, 현재 프로젝트 구현과도 가장 가깝다.
def systematic_resample(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return particle indices sampled with systematic resampling."""
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
    positions = (generator.random() + np.arange(n)) / n
    return np.searchsorted(cumulative, positions, side="left")
