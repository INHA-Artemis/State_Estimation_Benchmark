from __future__ import annotations

from pathlib import Path

import numpy as np



def normalize_bag_path(bag_path: Path) -> Path:
    bag_path = bag_path.expanduser()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path.parent
    return bag_path


def message_timestamp_ns(msg: object, fallback_timestamp: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_timestamp)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    return int(fallback_timestamp)


def sort_by_timestamp(timestamps: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(timestamps)
    sorted_items = [timestamps[order]]
    for array in arrays:
        sorted_items.append(array[order])
    return tuple(sorted_items)


def nearest_indices(reference_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference_timestamps, query_timestamps)
    indices = np.clip(indices, 1, len(reference_timestamps) - 1)
    prev_indices = indices - 1
    use_prev = np.abs(query_timestamps - reference_timestamps[prev_indices]) <= np.abs(
        query_timestamps - reference_timestamps[indices]
    )
    return np.where(use_prev, prev_indices, indices)


def estimate_linear_velocity(timestamps_ns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(positions, dtype=float)
    if len(positions) <= 1:
        return velocities
    dt = np.diff(timestamps_ns).astype(float) * 1e-9
    dt = np.clip(dt, 1e-9, None)
    velocities[1:] = np.diff(positions, axis=0) / dt[:, None]
    velocities[0] = velocities[1]
    return velocities


def build_noisy_position_measurements(
    gt: np.ndarray,
    dataset_cfg: dict,
    use_gt_key: str,
    default_enabled: bool = True,
) -> np.ndarray:
    if not bool(dataset_cfg.get(use_gt_key, default_enabled)):
        return np.zeros((len(gt), 3), dtype=float)

    seed = int(dataset_cfg.get("seed", 10))
    rng = np.random.default_rng(seed)
    default_std = np.array([0.7, 0.7, 0.7], dtype=float)
    meas_std = np.asarray(dataset_cfg.get("gnss_noise_std", default_std), dtype=float).reshape(-1)
    if meas_std.size == 1:
        meas_std = np.full(3, float(meas_std.item()), dtype=float)
    elif meas_std.size == 2:
        meas_std = np.array([meas_std[0], meas_std[1], meas_std[1]], dtype=float)
    elif meas_std.size > 3:
        meas_std = meas_std[:3]

    return gt[:, :3] + rng.normal(0.0, meas_std, size=(len(gt), 3))
