# IMU/GPS State Estimation Benchmark Library

This project provides a **Python-based benchmark library for IMU/GPS state estimation algorithms**.  
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
  - GPS on/off
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
2. Supports flexible sensor configurations (IMU/GPS on/off)  
3. Provides clear visualization and performance evaluation tools  
4. Enables easy comparison across multiple datasets  

## Particle Filter 실행 방법

현재 구현 기준으로 실제 실행 가능한 필터는 **PF**입니다.  
(`--filter ekf`, `--filter inekf`는 구조만 준비되어 있고 아직 구현 전입니다.)

### 1) Local 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_pf.py --mode fused
```

mode 예시:

```bash
python3 run_pf.py --mode imu_only
python3 run_pf.py --mode gps_only
python3 run_pf.py --mode fused
```

config 경로 override 예시:

```bash
python3 run_pf.py \
  --pf-config config/pf.yaml \
  --dataset-config config/dataset_config.yaml \
  --output-dir outputs/manual_run
```

실행 후 생성물(기본):
- `outputs/<filter>_<mode>_<timestamp>/run_data.npz`
- `outputs/<filter>_<mode>_<timestamp>/summary.json`
- `outputs/<filter>_<mode>_<timestamp>/trajectory.png`
- `outputs/<filter>_<mode>_<timestamp>/error.png`
- `outputs/<filter>_<mode>_<timestamp>/trajectory_animation.gif` (또는 mp4)

### 2) Docker 실행

이미지 빌드:

```bash
docker build -t state-estimation-benchmark:latest .
```

PF 실행 (결과를 로컬 `./outputs`에 저장):

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  state-estimation-benchmark:latest
```

mode 변경 실행:

```bash
docker run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  state-estimation-benchmark:latest \
  python run_pf.py --mode gps_only
```

## References

The implementation is inspired by existing libraries and tutorials:

- navlie  
- FilterPy  
- Stone Soup tutorials  
- robot_localization  
- DRIFT
