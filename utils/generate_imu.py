from __future__ import annotations

import numpy as np

from utils.rotation_utils import rpy_to_rot


def _wrap_angles(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(arr), np.cos(arr))


def _expand_motion_noise(values, pose_type: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("IMU noise and bias config must not be empty.")

    if pose_type == "2d":
        if arr.size == 1:
            return np.repeat(arr, 2)
        if arr.size == 2:
            return arr
        raise ValueError("For 2D synthetic IMU, use one shared value or [speed, yaw_rate].")

    if arr.size == 1:
        return np.repeat(arr, 6)
    if arr.size == 2:
        linear_std, angular_std = arr
        return np.array([linear_std, linear_std, linear_std, angular_std, angular_std, angular_std], dtype=float)
    if arr.size == 6:
        return arr
    raise ValueError("For 3D synthetic IMU, use one shared value, [linear, angular], or 6 axis-specific values.")


def _build_gt_2d(n: int, dt: float) -> np.ndarray:
    gt = np.zeros((n, 3), dtype=float)

    timeline = np.arange(n, dtype=float) * dt
    speed = 1.0 + 0.35 * np.sin(0.55 * timeline) + 0.15 * np.cos(0.15 * timeline)
    yaw_rate = 0.28 * np.sin(0.25 * timeline) + 0.08 * np.cos(0.07 * timeline)

    for k in range(1, n):
        x_prev, y_prev, yaw_prev = gt[k - 1]
        yaw_next = _wrap_angles(yaw_prev + yaw_rate[k] * dt)
        x_next = x_prev + speed[k] * np.cos(yaw_prev) * dt
        y_next = y_prev + speed[k] * np.sin(yaw_prev) * dt
        gt[k] = np.array([x_next, y_next, yaw_next], dtype=float)

    return gt


def _build_gt_3d(n: int, dt: float, initial_rpy_deg: np.ndarray) -> np.ndarray:
    gt = np.zeros((n, 6), dtype=float)
    gt[0, 3:6] = np.deg2rad(initial_rpy_deg)

    timeline = np.arange(n, dtype=float) * dt
    body_v = np.zeros((n, 3), dtype=float)
    body_v[:, 0] = 1.0 + 0.25 * np.sin(0.45 * timeline) + 0.1 * np.cos(0.11 * timeline)
    body_v[:, 1] = 0.4 * np.sin(0.31 * timeline + 0.4)
    body_v[:, 2] = 0.22 * np.cos(0.23 * timeline) - 0.08 * np.sin(0.17 * timeline)

    angular = np.zeros((n, 3), dtype=float)
    angular[:, 0] = 0.18 * np.sin(0.27 * timeline)
    angular[:, 1] = 0.14 * np.cos(0.19 * timeline + 0.4)
    angular[:, 2] = 0.22 * np.sin(0.15 * timeline) + 0.06 * np.cos(0.41 * timeline)

    for k in range(1, n):
        prev = gt[k - 1]
        rotation_world_from_body = rpy_to_rot(prev[3:6])
        world_velocity = rotation_world_from_body @ body_v[k]

        position_next = prev[0:3] + world_velocity * dt
        rpy_next = _wrap_angles(prev[3:6] + angular[k] * dt)

        gt[k, 0:3] = position_next
        gt[k, 3:6] = rpy_next

    return gt


def _controls_from_gt(gt: np.ndarray, pose_type: str, dt: float) -> np.ndarray:
    n = int(gt.shape[0])
    safe_dt = max(float(dt), 1e-12)

    if pose_type == "2d":
        controls = np.zeros((n, 2), dtype=float)
        if n <= 1:
            return controls

        delta_pos = gt[1:, 0:2] - gt[:-1, 0:2]
        controls[1:, 0] = np.linalg.norm(delta_pos, axis=1) / safe_dt
        controls[1:, 1] = _wrap_angles(gt[1:, 2] - gt[:-1, 2]) / safe_dt
        controls[0] = controls[1]
        return controls

    controls = np.zeros((n, 6), dtype=float)
    if n <= 1:
        return controls

    world_velocity = (gt[1:, 0:3] - gt[:-1, 0:3]) / safe_dt
    angular_rate = _wrap_angles(gt[1:, 3:6] - gt[:-1, 3:6]) / safe_dt

    for k in range(1, n):
        rotation_world_from_body = rpy_to_rot(gt[k - 1, 3:6])
        controls[k, 0:3] = rotation_world_from_body.T @ world_velocity[k - 1]
        controls[k, 3:6] = angular_rate[k - 1]

    controls[0] = controls[1]
    return controls


def _colored_noise(rng: np.random.Generator, noise_std: np.ndarray, rows: int, corr: float) -> np.ndarray:
    white = rng.normal(0.0, noise_std, size=(rows, noise_std.size))
    if rows == 0:
        return white

    corr = float(np.clip(corr, 0.0, 0.999))
    scale = float(np.sqrt(max(1.0 - corr * corr, 1e-12)))

    out = np.zeros_like(white)
    out[0] = white[0]
    for k in range(1, rows):
        out[k] = corr * out[k - 1] + scale * white[k]
    return out


def _integrate_controls(controls: np.ndarray, pose_type: str, dt: float, initial_state: np.ndarray) -> np.ndarray:
    n = int(controls.shape[0])
    safe_dt = float(dt)

    if pose_type == "2d":
        path = np.zeros((n, 3), dtype=float)
        if n == 0:
            return path
        path[0] = initial_state[0:3]

        for k in range(1, n):
            x_prev, y_prev, yaw_prev = path[k - 1]
            speed_k, yaw_rate_k = controls[k]
            yaw_next = _wrap_angles(yaw_prev + yaw_rate_k * safe_dt)
            x_next = x_prev + speed_k * np.cos(yaw_prev) * safe_dt
            y_next = y_prev + speed_k * np.sin(yaw_prev) * safe_dt
            path[k] = np.array([x_next, y_next, yaw_next], dtype=float)

        return path

    path = np.zeros((n, 6), dtype=float)
    if n == 0:
        return path
    path[0] = initial_state[0:6]

    for k in range(1, n):
        prev = path[k - 1]
        body_velocity = controls[k, 0:3]
        angular_rate = controls[k, 3:6]

        rotation_world_from_body = rpy_to_rot(prev[3:6])
        world_velocity = rotation_world_from_body @ body_velocity

        path[k, 0:3] = prev[0:3] + world_velocity * safe_dt
        path[k, 3:6] = _wrap_angles(prev[3:6] + angular_rate * safe_dt)

    return path


def _align_noisy_controls_to_gt(
    noisy_controls: np.ndarray,
    gt: np.ndarray,
    pose_type: str,
    dt: float,
    follow_gain: float,
    angular_follow_gain: float,
) -> np.ndarray:
    n = int(noisy_controls.shape[0])
    if n <= 1:
        return noisy_controls

    safe_dt = max(float(dt), 1e-12)
    follow_gain = float(max(follow_gain, 0.0))
    angular_follow_gain = float(max(angular_follow_gain, 0.0))

    if follow_gain == 0.0 and angular_follow_gain == 0.0:
        return noisy_controls

    adjusted = noisy_controls.copy()
    imu_path = _integrate_controls(adjusted, pose_type=pose_type, dt=safe_dt, initial_state=gt[0])

    for k in range(1, n):
        if pose_type == "2d":
            pos_error = gt[k - 1, 0:2] - imu_path[k - 1, 0:2]
            yaw = imu_path[k - 1, 2]
            forward_error = pos_error[0] * np.cos(yaw) + pos_error[1] * np.sin(yaw)
            adjusted[k, 0] += follow_gain * forward_error / safe_dt

            yaw_error = _wrap_angles(gt[k - 1, 2] - imu_path[k - 1, 2])
            adjusted[k, 1] += angular_follow_gain * yaw_error / safe_dt
            continue

        pos_error_world = gt[k - 1, 0:3] - imu_path[k - 1, 0:3]
        rotation_world_from_body = rpy_to_rot(imu_path[k - 1, 3:6])
        pos_error_body = rotation_world_from_body.T @ pos_error_world
        adjusted[k, 0:3] += follow_gain * pos_error_body / safe_dt

        rpy_error = _wrap_angles(gt[k - 1, 3:6] - imu_path[k - 1, 3:6])
        adjusted[k, 3:6] += angular_follow_gain * rpy_error / safe_dt

    return adjusted


def _sample_noisy_controls(ideal_controls: np.ndarray, dataset_cfg: dict, pose_type: str, dt: float, gt: np.ndarray) -> np.ndarray:
    seed = int(dataset_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    noise_std = _expand_motion_noise(dataset_cfg.get("imu_noise_std", 0.0), pose_type)
    bias_std = _expand_motion_noise(dataset_cfg.get("imu_bias_std", 0.0), pose_type)
    bias_rw_std = _expand_motion_noise(dataset_cfg.get("imu_bias_rw_std", 0.0), pose_type)

    fixed_bias = rng.normal(loc=0.0, scale=bias_std, size=ideal_controls.shape[1])

    noise_corr = float(dataset_cfg.get("imu_noise_correlation", 0.9))
    colored_noise = _colored_noise(rng, noise_std, ideal_controls.shape[0], corr=noise_corr)

    rw_step = rng.normal(0.0, bias_rw_std, size=ideal_controls.shape)
    rw_bias = np.cumsum(rw_step, axis=0) * np.sqrt(max(float(dt), 1e-12))

    noisy_controls = ideal_controls + fixed_bias[None, :] + colored_noise + rw_bias

# 이 값을 수정해서 imu_only mode의 비교 극대화
    follow_gain = float(dataset_cfg.get("imu_gt_follow_gain", 0.000)) # 0.005
    angular_follow_gain = float(dataset_cfg.get("imu_gt_angular_follow_gain", 0.000)) # 0.003
    noisy_controls = _align_noisy_controls_to_gt(
        noisy_controls,
        gt=gt,
        pose_type=pose_type,
        dt=dt,
        follow_gain=follow_gain,
        angular_follow_gain=angular_follow_gain,
    )

    return noisy_controls


def generate_imu_controls(dataset_cfg: dict, pose_type: str) -> tuple[np.ndarray, np.ndarray]:
    n = int(dataset_cfg.get("sequence_length", 300))
    dt = float(dataset_cfg.get("dt", 0.1))

    if pose_type == "2d":
        gt = _build_gt_2d(n=n, dt=dt)
        ideal_controls = _controls_from_gt(gt, pose_type=pose_type, dt=dt)
        noisy_controls = _sample_noisy_controls(ideal_controls, dataset_cfg, pose_type=pose_type, dt=dt, gt=gt)
        return noisy_controls, gt

    initial_rpy_deg = np.asarray(dataset_cfg.get("initial_rpy_deg", [10.0, -8.0, 20.0]), dtype=float).reshape(-1)
    if not initial_rpy_deg.size == 3:
        raise ValueError("initial_rpy_deg must contain exactly 3 values: [roll, pitch, yaw].")

    gt = _build_gt_3d(n=n, dt=dt, initial_rpy_deg=initial_rpy_deg)
    ideal_controls = _controls_from_gt(gt, pose_type=pose_type, dt=dt)
    noisy_controls = _sample_noisy_controls(ideal_controls, dataset_cfg, pose_type=pose_type, dt=dt, gt=gt)
    return noisy_controls, gt
