from typing import Optional, Sequence

import numpy as np


def h(x_t, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    측정 함수 h(x_t).
    - indices가 있으면 해당 상태 성분만 측정값으로 사용
    - indices가 없으면 상태 차원에 맞는 기본 위치 성분을 사용
      * 2D state (len=3): [x, y]
      * 3D/IMU state (len=6 or 9): [x, y, z]
    """
    x_t = np.asarray(x_t, dtype=float).reshape(-1)

    if indices is None:
        if x_t.size == 3:
            indices = [0, 1]
        elif x_t.size in (6, 9):
            indices = [0, 1, 2]
        else:
            raise ValueError(
                "indices is required for this state size. "
                "Supported defaults are state length 3, 6, or 9."
            )

    return x_t[np.asarray(indices, dtype=int)]


def measure(
    x_t,
    indices: Optional[Sequence[int]] = None,
    v_t=None,
    measurement_cov=None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    측정 모델:
        z_t = h(x_t) + v_t
    """
    z_nominal = h(x_t, indices=indices)
    dim = z_nominal.size

    if v_t is not None:
        noise = np.asarray(v_t, dtype=float).reshape(-1)
        if noise.size != dim:
            raise ValueError("v_t size must match measurement size.")
    elif measurement_cov is not None:
        cov = np.asarray(measurement_cov, dtype=float)
        if cov.shape != (dim, dim):
            raise ValueError("measurement_cov must be shape (meas_dim, meas_dim).")
        generator = np.random.default_rng() if rng is None else rng
        noise = generator.multivariate_normal(np.zeros(dim), cov)
    else:
        noise = np.zeros(dim, dtype=float)

    return z_nominal + noise