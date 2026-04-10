import numpy as np

from utils.math_utils import exp_so3
from utils.rotation_utils import rot_to_rpy, rpy_to_rot


# The State Transition Function, f(x_{t-1}, u_t).
def f(
    x_prev,
    u_t,
    dt: float,
    g: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=float),
) -> np.ndarray:
    """
    상태 전이 함수 f(x_{t-1}, u_t)

    Supported state layouts:
    - length 9  : [p(3), v(3), rpy(3)]
    - length 15 : [p(3), v(3), rpy(3), b_a(3), b_w(3)]

    IMU input: [a(3), w(3)]
    1) R_t = R_{t-1} * exp((w_t - b_w) * dt)
    2) v_t = v_{t-1} + (R_{t-1}(a_t - b_a) + g) * dt
    3) p_t = p_{t-1} + v_{t-1} * dt + 0.5 * (R_{t-1}(a_t - b_a) + g) * dt^2
    4) bias는 random-walk 항이 없으므로 유지
    """
    x_prev = np.asarray(x_prev, dtype=float).reshape(-1)
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)

    if x_prev.size not in (9, 15):
        raise ValueError("x_prev must be length 9 or 15.")
    if u_t.size != 6:
        raise ValueError("u_t must be length 6: [ax, ay, az, wx, wy, wz].")
    if g.size != 3:
        raise ValueError("g must be length 3.")

    # 상태 분해
    p_prev = x_prev[0:3]
    v_prev = x_prev[3:6]
    rpy_prev = x_prev[6:9]
    if x_prev.size == 15:
        b_a = x_prev[9:12]
        b_w = x_prev[12:15]
    else:
        b_a = np.zeros(3, dtype=float)
        b_w = np.zeros(3, dtype=float)

    # 입력 분해: 선가속도, 각속도
    a_t = u_t[0:3] - b_a
    w_t = u_t[3:6] - b_w

    # (1) orientation update
    R_prev = rpy_to_rot(rpy_prev)
    R_t = R_prev @ exp_so3(w_t * dt)

    # (2) velocity update
    accel_world = R_prev @ a_t + g
    v_t = v_prev + accel_world * dt

    # (3) position update
    p_t = p_prev + v_prev * dt + 0.5 * accel_world * (dt**2)
    rpy_t = rot_to_rpy(R_t)

    if x_prev.size == 9:
        return np.concatenate([p_t, v_t, rpy_t])
    return np.concatenate([p_t, v_t, rpy_t, b_a, b_w])
