import numpy as np

# 상태 벡터 라벨 정의
# 2d: [x, y, yaw], 3d: [x, y, z, roll, pitch, yaw]
STATE_LABELS = {
    "2d": ("x", "y", "yaw"),
    "3d": ("x", "y", "z", "roll", "pitch", "yaw"),
}


def state_dim(pose_type):
    # 상태 벡터 차원 수 반환
    if pose_type not in STATE_LABELS:
        raise ValueError("pose_type must be '2d' or '3d'.")
    return len(STATE_LABELS[pose_type])


def zero_state(pose_type):
    # 0으로 초기화된 상태 벡터 생성
    return np.zeros(state_dim(pose_type), dtype=float)


def state_vector(values, pose_type):
    # 입력 값을 1차원 numpy 상태 벡터로 변환하고 크기 검증
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != state_dim(pose_type):
        raise ValueError("State vector size does not match pose_type.")
    return vector
