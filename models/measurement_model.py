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
