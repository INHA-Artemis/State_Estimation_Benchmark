from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset_loader_utils import (
    build_noisy_position_measurements,
    estimate_linear_velocity,
    message_timestamp_ns,
    nearest_indices,
    normalize_bag_path,
    sort_by_timestamp,
)
from utils.rotation_utils import quat_to_rpy

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

    bag_path = normalize_bag_path(Path(dataset_cfg["rosbag_path"]))
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

    imu_indices = nearest_indices(imu_timestamps, gt_timestamps)
    gyro_samples = gyro[imu_indices]

    if linear_source == "gt_velocity":
        linear_samples = estimate_linear_velocity(gt_timestamps, gt[:, :3])
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

    measurements = build_noisy_position_measurements(gt, dataset_cfg, "rosbag_use_gt_as_gnss")
    return controls, measurements, gt, dt, gt_timestamps


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
            msg_timestamp = message_timestamp_ns(msg, timestamp)
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

    return sort_by_timestamp(
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
            msg_timestamp = message_timestamp_ns(msg, timestamp)
            rpy = quat_to_rpy(
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

    timestamps_np, gt_np = sort_by_timestamp(np.asarray(timestamps, dtype=np.int64), np.asarray(gt_rows, dtype=float))
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
