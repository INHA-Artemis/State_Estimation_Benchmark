# IMU/GNSS State Estimation Benchmark Library

This project provides a **Python-based benchmark library for IMU/GNSS state estimation algorithms**.  
The goal is to implement and compare multiple filtering methods under a **unified and modular interface**.

## Features

- Implementation of multiple state estimation algorithms:
  - Extended Kalman Filter (**EKF**)
  - Unscented Kalman Filter (**UKF**)
  - Particle Filter (**PF**)
  - Invariant Extended Kalman Filter (**InEKF**)

- **Unified interface** for all filters
- Easy configuration:
  - IMU on/off
  - GNSS on/off
  - Filter parameter tuning
- **Real-time estimation visualization**
- **Performance analysis and comparison across datasets**

## State Models

The library supports two state representations:

- **3D model**
  
  \[
  (x, y, \psi)
  \]

- **6D model**

  Full 6-DoF state estimation.

## Goal

The objective of this project is to build a **modular Python benchmark library** that:

1. Implements EKF, UKF, PF, and InEKF under a common interface  
2. Supports flexible sensor configurations (IMU/GNSS on/off)  
3. Provides clear visualization and performance evaluation tools  
4. Enables easy comparison across multiple datasets  

## Reference Comparison

기존 대표 레퍼런스들과 현재 프로젝트 구현 범위를 비교하면 아래와 같습니다.

| 항목 \ Reference | [navlie](https://github.com/decargroup/navlie) | [FilterPy](https://github.com/rlabbe/filterpy) | [Stone Soup](https://stonesoup.readthedocs.io/en/v1.2/auto_tutorials/index.html) | [robot_localization](https://www.notion.so/4-Robot-Localization-31e5215b8741803cba0fc205c165a59e) | [DRIFT](https://github.com/UMich-CURLY/drift) | OURS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Python 기반** | O | O | O | X | X | O |
| **EKF** | O | O | O | O | X | O |
| **UKF** | O | O | O | O | X | O |
| **PF** | X | O | O | X | X | O |
| **InEKF** | X | X | X | X | O | O |
| **실시간 estimation visualization** | X | X | X | X | X | O |
| **IMU 관련 모델/입력 지원** | O | X | X | O | O | O |
| **GPS 관련 처리/연동** | X | X | X | O | X | O |
| **IMU on/off** | X | X | X | X | X | O |
| **GPS on/off** | X | X | X | X | X | O |


## Supported Datasets

현재 repository에서 지원하는 데이터셋은 아래와 같습니다.

- `EuRoC`: available
- `rosbag`: available
  - `ROS1 .bag`: available
  - `ROS2 bag (directory / metadata.yaml / .db3)`: available
  - [KAIST VIO](https://github.com/url-kaist/kaistviodataset/tree/main): available
  - [M2DGR](https://github.com/SJTU-ViSYS/M2DGR?tab=readme-ov-file#dataset-sequences): available
- `KITTI`: TBD
- `TUM`: TBD


## Particle Filter 실행 방법

현재 구현 기준으로 실제 실행 가능한 필터는 **PF**입니다.  
(`--filter ekf`, `--filter inekf`는 구조만 준비되어 있고 아직 구현 전입니다.)

### 1) Local 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 examples/run_pf.py
```

config 경로 override 예시:

```bash
python3 examples/run_pf.py \
  --dataset-config config/dataset_config.yaml \
  --pf-config config/pf.yaml \
  --output-dir outputs
```

실행 후 콘솔 출력(기본):
- `Pose type`
- `Dataset CSV`
- `Steps`
- `RMSE (position)`
- `Plot saved`
- `Error plot saved`
- `Animation saved` 또는 `Animation skipped`

실행 후 생성물(기본):
- `outputs/<dataset_name>_dataset.csv` 또는 `generated_csv_path`에 지정한 통합 dataset CSV
- `outputs/<dataset_name>_pf_estimates.csv`
- `outputs/<dataset_name>_pf_trajectory.png`
- `outputs/<dataset_name>_pf_position_error_norm.png`
- `outputs/<dataset_name>_pf_trajectory.mp4` 또는 `outputs/<dataset_name>_pf_trajectory.gif`  
  단, `pf.yaml`에서 `visualization.save_animation: true` 이어야 하며,
  mp4 저장에는 `ffmpeg`, gif 저장에는 `Pillow` writer가 필요합니다.

### 1-1) rosbag 사용 방법

`dataset_config.yaml`에서 아래 항목만 맞게 바꾸면 새로운 rosbag을 바로 연결할 수 있습니다.

```yaml
dataset_type: rosbag
dataset_name: kaistVio
rosbag_path: /path/to/your_rosbag
rosbag_imu_topic: /mavros/imu/data
rosbag_gt_topic: /pose_transformed
rosbag_linear_source: gt_velocity  # or accel
rosbag_use_gt_as_gnss: true
```

설정 가이드:
- `dataset_type`: `rosbag`으로 설정
- `dataset_name`: 출력 파일명에 사용될 이름
- `rosbag_path`:
  - ROS1: `.bag` 파일 경로
  - ROS2: bag 디렉터리, `metadata.yaml`, 또는 `.db3` 경로
- `rosbag_imu_topic`: `sensor_msgs/Imu` topic 이름
- `rosbag_gt_topic`: pose-like GT topic 이름
- `rosbag_linear_source`:
  - `gt_velocity`: GT position 차분으로 선형 속도 생성
  - `accel`: raw accelerometer 사용

현재 loader가 기대하는 메시지 형태:
- IMU: `linear_acceleration`, `angular_velocity` 필드가 있는 `sensor_msgs/Imu`
- GT: `PoseStamped`, `PoseWithCovarianceStamped`, `Odometry.pose`, `TransformStamped`

예를 들어 KAIST VIO Dataset의 ROS1 bag를 사용할 때는 아래처럼 둘 수 있습니다.

```yaml
dataset_type: rosbag
dataset_name: kaistVio
rosbag_path: /kaistvio_dataset/circle.bag
rosbag_imu_topic: /mavros/imu/data
rosbag_gt_topic: /pose_transformed
rosbag_linear_source: gt_velocity
```

### PF Resampling Reference

PF resampling 알고리즘(multinomial, residual, systematic, stratified) 개념을 정리할 때는 아래 자료를 참고할 수 있습니다.

- Roger Labbe, *Kalman and Bayesian Filters in Python*, Chapter 12 Particle Filters:  
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

### EuRoC 결과 비교 표

동일한 데이터셋 조건으로 비교합니다.
- Dataset: `EuRoC MAV`
- Sequence: `V1_01_easy`
- Sensor basis: `imu`
- Environment: `Vicon Room 1`

| Filter | RMSE (position) | Runtime (filter only) | Figure | Video | Note |
| :---: | :---: | :---: | :---: | :---: | :---: |
| KF | TBD | TBD | `outputs/kf_trajectory_3d.png` | `outputs/kf_trajectory_3d.mp4` | - |
| UKF | TBD | TBD | `outputs/ukf_trajectory_3d.png` | `outputs/ukf_trajectory_3d.mp4` | - |
| PF | `0.3218` | `8.563 sec` | `outputs/pf_trajectory_3d.png` | `outputs/pf_trajectory_3d.mp4` | `Samples 2000` |
| InEKF | TBD | TBD | `outputs/inekf_trajectory_3d.png` | `outputs/inekf_trajectory_3d.mp4` | - |

### 2) Docker 실행

이미지 빌드:

```bash
docker build -t state-estimation-benchmark:latest .
```

PF 실행 (결과를 로컬 `./outputs`에 저장):

```bash
docker run --rm   -v "$(pwd)/outputs:/app/outputs"   state-estimation-benchmark:latest
```

mode 변경 실행:

```bash
docker run --rm   -v "$(pwd)/outputs:/app/outputs"   state-estimation-benchmark:latest   python run_pf.py --mode gnss_only
```

## References

The implementation is inspired by existing libraries and tutorials:

- [navlie](https://github.com/decargroup/navlie)  
- [FilterPy](https://github.com/rlabbe/filterpy)  
- [Stone Soup tutorials](https://stonesoup.readthedocs.io/en/v1.2/auto_tutorials/index.html)  
- [robot_localization](https://www.notion.so/4-Robot-Localization-31e5215b8741803cba0fc205c165a59e)  
- [DRIFT](https://github.com/UMich-CURLY/drift)  
- Roger Labbe, *Kalman and Bayesian Filters in Python*, Chapter 12 Particle Filters  
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
