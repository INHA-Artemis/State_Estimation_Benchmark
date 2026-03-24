from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.rotation_utils import _rot_to_rpy

try:
    from rosbags.highlevel import AnyReader
except ImportError:  # pragma: no cover
    AnyReader = None


def load_rosbag_dataset(dataset_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if AnyReader is None:
        raise ImportError(
            "The 'rosbags' package is required to read ROS bag files. "
            "Install it with `pip install rosbags`."
        )

    bag_path = _normalize_bag_path(Path(dataset_cfg["rosbag_path"]))
    imu_topic = str(dataset_cfg.get("rosbag_imu_topic", "/mavros/imu/data"))
    gt_topic = str(dataset_cfg.get("rosbag_gt_topic", "/pose_transformed"))
    linear_source = str(dataset_cfg.get("rosbag_linear_source", "accel")).strip().lower()

    imu_timestamps, accel, gyro = _load_imu_messages(bag_path, imu_topic)
    gt_timestamps, gt = _load_gt_messages(bag_path, gt_topic)

    overlap_start = max(int(imu_timestamps[0]), int(gt_timestamps[0]))
    overlap_end = min(int(imu_timestamps[-1]), int(gt_timestamps[-1]))
    valid_mask = (gt_timestamps >= overlap_start) & (gt_timestamps <= overlap_end)
    if not np.any(valid_mask):
        raise ValueError("ROS bag IMU and ground-truth topics do not share an overlapping time range.")

    gt_timestamps = gt_timestamps[valid_mask]
    gt = gt[valid_mask]

    imu_indices = _nearest_indices(imu_timestamps, gt_timestamps)
    gyro_samples = gyro[imu_indices]

    if linear_source == "gt_velocity":
        linear_samples = _estimate_linear_velocity(gt_timestamps, gt[:, :3])
    elif linear_source == "accel":
        linear_samples = accel[imu_indices]
    else:
        raise ValueError("rosbag_linear_source must be either 'accel' or 'gt_velocity'.")

    controls = np.zeros((len(gt_timestamps), 6), dtype=float)
    controls[:, :3] = linear_samples
    controls[:, 3:] = gyro_samples

    dt = np.zeros(len(gt_timestamps), dtype=float)
    if len(gt_timestamps) > 1:
        dt[1:] = np.diff(gt_timestamps) * 1e-9
        dt[0] = dt[1]
    else:
        dt[0] = float(dataset_cfg.get("dt", 0.005))

    measurements = _build_position_measurements(gt, dataset_cfg)
    return controls, measurements, gt, dt, gt_timestamps


def _normalize_bag_path(bag_path: Path) -> Path:
    bag_path = bag_path.expanduser()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    # ROS2 bag inputs may be passed as the bag directory, metadata.yaml, or a .db3 file.
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path.parent
    return bag_path


def _load_imu_messages(bag_path: Path, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    accel_rows: list[list[float]] = []
    gyro_rows: list[list[float]] = []

    with AnyReader([bag_path]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        if not connections:
            available = sorted({x.topic for x in reader.connections})
            raise ValueError(f"IMU topic '{topic}' not found in bag. Available topics: {available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            msg_timestamp = _message_timestamp_ns(msg, timestamp)
            timestamps.append(msg_timestamp)
            accel_rows.append(
                [
                    float(msg.linear_acceleration.x),
                    float(msg.linear_acceleration.y),
                    float(msg.linear_acceleration.z),
                ]
            )
            gyro_rows.append(
                [
                    float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y),
                    float(msg.angular_velocity.z),
                ]
            )

    if not timestamps:
        raise ValueError(f"No IMU messages found on topic '{topic}'.")

    return _sort_by_timestamp(
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(accel_rows, dtype=float),
        np.asarray(gyro_rows, dtype=float),
    )


def _load_gt_messages(bag_path: Path, topic: str) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    gt_rows: list[list[float]] = []

    with AnyReader([bag_path]) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        if not connections:
            available = sorted({x.topic for x in reader.connections})
            raise ValueError(f"Ground-truth topic '{topic}' not found in bag. Available topics: {available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            pose = _extract_pose(msg)
            msg_timestamp = _message_timestamp_ns(msg, timestamp)
            rpy = _quat_to_rpy(
                float(pose["orientation"][3]),
                float(pose["orientation"][0]),
                float(pose["orientation"][1]),
                float(pose["orientation"][2]),
            )
            timestamps.append(msg_timestamp)
            gt_rows.append(
                [
                    float(pose["position"][0]),
                    float(pose["position"][1]),
                    float(pose["position"][2]),
                    float(rpy[0]),
                    float(rpy[1]),
                    float(rpy[2]),
                ]
            )

    if not timestamps:
        raise ValueError(f"No ground-truth messages found on topic '{topic}'.")

    timestamps_np, gt_np = _sort_by_timestamp(np.asarray(timestamps, dtype=np.int64), np.asarray(gt_rows, dtype=float))
    return timestamps_np, gt_np


def _extract_pose(msg: Any) -> dict[str, np.ndarray]:
    if hasattr(msg, "pose"):
        pose = msg.pose
        if hasattr(pose, "pose"):
            pose = pose.pose
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            return {
                "position": np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float),
                "orientation": np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float),
            }

    if hasattr(msg, "transform"):
        transform = msg.transform
        return {
            "position": np.array([transform.translation.x, transform.translation.y, transform.translation.z], dtype=float),
            "orientation": np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w], dtype=float),
        }

    if all(hasattr(msg, attr) for attr in ("position", "orientation")):
        return {
            "position": np.array([msg.position.x, msg.position.y, msg.position.z], dtype=float),
            "orientation": np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=float),
        }

    raise TypeError("Unsupported ground-truth message type. Expected a pose-like ROS message.")


def _message_timestamp_ns(msg: Any, fallback_timestamp: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_timestamp)

    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)

    return int(fallback_timestamp)


def _sort_by_timestamp(timestamps: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(timestamps)
    sorted_items = [timestamps[order]]
    for array in arrays:
        sorted_items.append(array[order])
    return tuple(sorted_items)


def _nearest_indices(reference_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference_timestamps, query_timestamps)
    indices = np.clip(indices, 1, len(reference_timestamps) - 1)
    prev_indices = indices - 1
    use_prev = np.abs(query_timestamps - reference_timestamps[prev_indices]) <= np.abs(
        query_timestamps - reference_timestamps[indices]
    )
    return np.where(use_prev, prev_indices, indices)


def _estimate_linear_velocity(timestamps_ns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(positions, dtype=float)
    if len(positions) <= 1:
        return velocities

    dt = np.diff(timestamps_ns).astype(float) * 1e-9
    dt = np.clip(dt, 1e-9, None)
    velocities[1:] = np.diff(positions, axis=0) / dt[:, None]
    velocities[0] = velocities[1]
    return velocities


def _quat_to_rpy(w: float, x: float, y: float, z: float) -> np.ndarray:
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return _rot_to_rpy(rotation)


def _build_position_measurements(gt: np.ndarray, dataset_cfg: dict) -> np.ndarray:
    if not bool(dataset_cfg.get("rosbag_use_gt_as_gnss", True)):
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
