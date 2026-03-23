from typing import Optional

import numpy as np

from utils.math_utils import _exp_so3
from utils.rotation_utils import _rot_to_rpy, _rpy_to_rot

# The State Transition Function, f(x_{t-1}, u_t).
def f(
    x_prev,
    u_t,
    dt: float,
    g: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=float),
) -> np.ndarray:
    """
    상태 전이 함수 f(x_{t-1}, u_t)
    IMU motion model (state: [p(3), v(3), rpy(3)], input: [a(3), w(3)]).
    1) R_t = R_{t-1} * exp(w_t * dt)
    2) v_t = v_{t-1} + (R_{t-1} a_t + g) * dt
    3) p_t = p_{t-1} + v_{t-1} * dt + 0.5 * (R_{t-1} a_t + g) * dt^2
    """
    x_prev = np.asarray(x_prev, dtype=float).reshape(-1)
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)

    if x_prev.size != 9:
        raise ValueError("x_prev must be length 9: [px,py,pz,vx,vy,vz,roll,pitch,yaw].")
    if u_t.size != 6:
        raise ValueError("u_t must be length 6: [ax,ay,az,wx,wy,wz].")
    if g.size != 3:
        raise ValueError("g must be length 3.")

    # 상태 분해: 위치, 속도, 자세(RPY)
    p_prev = x_prev[0:3]
    v_prev = x_prev[3:6]
    rpy_prev = x_prev[6:9]

    # 입력 분해: 선가속도, 각속도
    a_t = u_t[0:3]
    w_t = u_t[3:6]

    # (1) orientation update
    R_prev = _rpy_to_rot(rpy_prev)
    R_t = R_prev @ _exp_so3(w_t * dt)

    # (2) velocity update
    accel_world = R_prev @ a_t + g
    v_t = v_prev + accel_world * dt
    # (3) position update
    p_t = p_prev + v_prev * dt + 0.5 * accel_world * (dt**2)
    rpy_t = _rot_to_rpy(R_t)

    return np.concatenate([p_t, v_t, rpy_t])


def propagate(
    x_prev,
    u_t,
    dt: float,
    v_t=None,
    process_cov=None,
    rng: Optional[np.random.Generator] = None,
    g: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=float),
) -> np.ndarray:
    """
    최종 예측식: x_t = f(x_{t-1}, u_t) + v_t
    """
    x_nominal = f(x_prev, u_t, dt=dt, g=g)
    dim = x_nominal.size

    # 프로세스 노이즈 처리 우선순위:
    # 1) v_t 직접 입력
    # 2) process_cov 기반 샘플링
    # 3) 노이즈 0
    if v_t is not None:
        noise = np.asarray(v_t, dtype=float).reshape(-1)
        if noise.size != dim:
            raise ValueError("v_t size must match state size.")
    elif process_cov is not None:
        cov = np.asarray(process_cov, dtype=float)
        if cov.shape != (dim, dim):
            raise ValueError("process_cov must be shape (state_dim, state_dim).")
        generator = np.random.default_rng() if rng is None else rng
        noise = generator.multivariate_normal(np.zeros(dim), cov)
    else:
        noise = np.zeros(dim, dtype=float)

    return x_nominal + noise
