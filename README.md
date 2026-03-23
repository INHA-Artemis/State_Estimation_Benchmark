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
python3 examples/run_pf.py   --dataset-config config/dataset_config.yaml   --pf-config config/pf.yaml   --output-dir outputs
```

실행 후 콘솔 출력(기본):
- `Pose type`
- `Dataset CSV`
- `Steps`
- `RMSE (position)`
- `Plot saved`
- `Animation saved` 또는 `Animation skipped`

실행 후 생성물(기본):
- `outputs/euroc_6d.csv` 또는 `generated_csv_path`에 지정한 통합 dataset CSV
- `outputs/pf_trajectory_2d.png` 또는 `outputs/pf_trajectory_3d.png`
- `outputs/pf_trajectory_2d.mp4` 또는 `outputs/pf_trajectory_3d.mp4`  
  단, `pf.yaml`에서 `visualization.save_animation: true` 이고 ffmpeg가 있어야 저장됩니다.

### PF Resampling Reference

PF resampling 알고리즘(multinomial, residual, systematic, stratified) 개념을 정리할 때는 아래 자료를 참고할 수 있습니다.

- Roger Labbe, *Kalman and Bayesian Filters in Python*, Chapter 12 Particle Filters:  
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

### EuRoC 결과 비교 표

동일한 EuRoC 데이터셋에 대해 필터별 성능을 비교할 때는 아래 표를 채워서 사용할 수 있습니다.
기준 데이터셋은 `EuRoC MAV`, 시퀀스는 `V1_01_easy`, 센서는 `imu` 기준입니다.

| Filter | Dataset | RMSE (position) | Runtime (filter only) | Figure | Video |
| :---: | :---: | :---: | :---: | :---: | :---: |
| KF | EuRoC Vicon Room 1 / V1_01_easy (imu) | TBD | TBD | `outputs/kf_trajectory_3d.png` | `outputs/kf_trajectory_3d.mp4` |
| UKF | EuRoC Vicon Room 1 / V1_01_easy (imu) | TBD | TBD | `outputs/ukf_trajectory_3d.png` | `outputs/ukf_trajectory_3d.mp4` |
| PF | EuRoC Vicon Room 1 / V1_01_easy (imu) | `0.3218` | `8.563 sec` | `outputs/pf_trajectory_3d.png` | `outputs/pf_trajectory_3d.mp4` |
| InEKF | EuRoC Vicon Room 1 / V1_01_easy (imu) | TBD | TBD | `outputs/inekf_trajectory_3d.png` | `outputs/inekf_trajectory_3d.mp4` |

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
