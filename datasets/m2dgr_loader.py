from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.rotation_utils import _rot_to_rpy

try:
    from rosbags.highlevel import AnyReader
except ImportError:  # pragma: no cover
    AnyReader = None


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def load_m2dgr_dataset(dataset_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if AnyReader is None:
        raise ImportError(
            "The 'rosbags' package is required to read ROS bag files. "
            "Install it with `pip install rosbags`."
        )

    bag_path = _normalize_bag_path(Path(dataset_cfg["m2dgr_bag_path"]))
    gt_txt_path = Path(dataset_cfg["m2dgr_gt_txt_path"]).expanduser()
    imu_topic = str(dataset_cfg.get("m2dgr_imu_topic", "/handsfree/imu"))
    gnss_topic = str(dataset_cfg.get("m2dgr_gnss_topic", "/ublox/fix"))
    linear_source = str(dataset_cfg.get("m2dgr_linear_source", "gt_velocity")).strip().lower()

    if not gt_txt_path.exists():
        raise FileNotFoundError(f"M2DGR GT txt path does not exist: {gt_txt_path}")

    gt_timestamps, gt = _load_ground_truth_txt(gt_txt_path)
    imu_timestamps, accel, gyro = _load_imu_messages(bag_path, imu_topic)
    gnss_timestamps, gnss_measurements = _load_gnss_messages(bag_path, gnss_topic, gt)

    overlap_start = max(int(imu_timestamps[0]), int(gt_timestamps[0]))
    overlap_end = min(int(imu_timestamps[-1]), int(gt_timestamps[-1]))
    valid_mask = (gt_timestamps >= overlap_start) & (gt_timestamps <= overlap_end)
    if not np.any(valid_mask):
        raise ValueError("M2DGR IMU and ground-truth files do not share an overlapping time range.")

    gt_timestamps = gt_timestamps[valid_mask]
    gt = gt[valid_mask]

    imu_indices = _nearest_indices(imu_timestamps, gt_timestamps)
    gyro_samples = gyro[imu_indices]

    if linear_source == "gt_velocity":
        linear_samples = _estimate_linear_velocity(gt_timestamps, gt[:, :3])
    elif linear_source == "accel":
        linear_samples = accel[imu_indices]
    else:
        raise ValueError("m2dgr_linear_source must be either 'accel' or 'gt_velocity'.")

    controls = np.zeros((len(gt_timestamps), 6), dtype=float)
    controls[:, :3] = linear_samples
    controls[:, 3:] = gyro_samples

    dt = np.zeros(len(gt_timestamps), dtype=float)
    if len(gt_timestamps) > 1:
        dt[1:] = np.diff(gt_timestamps) * 1e-9
        dt[0] = dt[1]
    else:
        dt[0] = float(dataset_cfg.get("dt", 0.01))

    measurements = _build_measurements(gt_timestamps, gt, gnss_timestamps, gnss_measurements, dataset_cfg)
    return controls, measurements, gt, dt, gt_timestamps


def _normalize_bag_path(bag_path: Path) -> Path:
    bag_path = bag_path.expanduser()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path.parent
    return bag_path


def _load_ground_truth_txt(gt_txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    positions_ecef: list[list[float]] = []
    quaternions_xyzw: list[list[float]] = []

    with gt_txt_path.open("r", encoding="utf-8") as gt_file:
        for line in gt_file:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            timestamp_sec = float(parts[0])
            timestamps.append(int(round(timestamp_sec * 1e9)))
            positions_ecef.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quaternions_xyzw.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

    if not timestamps:
        raise ValueError(f"No GT rows found in {gt_txt_path}.")

    positions_ecef_np = np.asarray(positions_ecef, dtype=float)
    origin_lat, origin_lon, origin_alt = _ecef_to_lla(*positions_ecef_np[0])
    gt_positions = np.vstack([_ecef_to_enu(x, y, z, origin_lat, origin_lon, origin_alt) for x, y, z in positions_ecef_np])

    gt_rows = []
    for pos, quat_xyzw in zip(gt_positions, quaternions_xyzw):
        rpy = _quat_to_rpy(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
        gt_rows.append([float(pos[0]), float(pos[1]), float(pos[2]), float(rpy[0]), float(rpy[1]), float(rpy[2])])

    return np.asarray(timestamps, dtype=np.int64), np.asarray(gt_rows, dtype=float)


def _load_imu_messages(bag_path: Path, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    accel_rows: list[list[float]] = []
    gyro_rows: list[list[float]] = []

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise ValueError(f"IMU topic '{topic}' not found in bag. Available topics: {available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamps.append(_message_timestamp_ns(msg, timestamp))
            accel_rows.append([
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
            ])
            gyro_rows.append([
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ])

    if not timestamps:
        raise ValueError(f"No IMU messages found on topic '{topic}'.")

    return _sort_by_timestamp(
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(accel_rows, dtype=float),
        np.asarray(gyro_rows, dtype=float),
    )


def _load_gnss_messages(bag_path: Path, topic: str, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    lla_rows: list[list[float]] = []

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise ValueError(f"GNSS topic '{topic}' not found in bag. Available topics: {available}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamps.append(_message_timestamp_ns(msg, timestamp))
            lla_rows.append([float(msg.latitude), float(msg.longitude), float(msg.altitude)])

    if not timestamps:
        raise ValueError(f"No GNSS messages found on topic '{topic}'.")

    lla_np = np.asarray(lla_rows, dtype=float)
    origin_lat, origin_lon, origin_alt = lla_np[0]
    enu_rows = np.vstack([_lla_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt) for lat, lon, alt in lla_np])

    # Align the GNSS local frame to the GT local frame at the nearest timestamp.
    gnss_timestamps = np.asarray(timestamps, dtype=np.int64)
    gt_origin_idx = _nearest_indices(np.asarray(timestamps, dtype=np.int64), np.asarray([gnss_timestamps[0]], dtype=np.int64))[0]
    frame_offset = gt[min(gt_origin_idx, len(gt) - 1), :3] - enu_rows[0]
    enu_rows = enu_rows + frame_offset[None, :]

    return gnss_timestamps, enu_rows


def _build_measurements(
    gt_timestamps: np.ndarray,
    gt: np.ndarray,
    gnss_timestamps: np.ndarray,
    gnss_measurements: np.ndarray,
    dataset_cfg: dict,
) -> np.ndarray:
    if bool(dataset_cfg.get("m2dgr_use_gt_as_gnss", False)):
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

    gnss_indices = _nearest_indices(gnss_timestamps, gt_timestamps)
    return gnss_measurements[gnss_indices]


def _message_timestamp_ns(msg: object, fallback_timestamp: int) -> int:
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


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z], dtype=float)


def _ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    b = WGS84_A * np.sqrt(1.0 - WGS84_E2)
    ep = np.sqrt((WGS84_A**2 - b**2) / b**2)
    p = np.sqrt(x * x + y * y)
    th = np.arctan2(WGS84_A * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep * ep * b * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def _ecef_to_enu(x: float, y: float, z: float, origin_lat_deg: float, origin_lon_deg: float, origin_alt: float) -> np.ndarray:
    origin_ecef = _lla_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt)
    return _ecef_delta_to_enu(np.array([x, y, z], dtype=float) - origin_ecef, origin_lat_deg, origin_lon_deg)


def _lla_to_enu(lat_deg: float, lon_deg: float, alt: float, origin_lat_deg: float, origin_lon_deg: float, origin_alt: float) -> np.ndarray:
    ecef = _lla_to_ecef(lat_deg, lon_deg, alt)
    origin_ecef = _lla_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt)
    return _ecef_delta_to_enu(ecef - origin_ecef, origin_lat_deg, origin_lon_deg)


def _ecef_delta_to_enu(delta_ecef: np.ndarray, origin_lat_deg: float, origin_lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(origin_lat_deg)
    lon = np.deg2rad(origin_lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )
    return rot @ delta_ecef
