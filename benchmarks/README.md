# KAIST VIO Benchmark Input Flow

이 폴더의 benchmark script들은 KAIST VIO rosbag을 바로 filter에 넣지 않고, 먼저 공통 dataset 형태로 변환한 뒤 각 filter/repo가 기대하는 입력으로 다시 맞춘다.

기본 rosbag 입력은 다음 topic을 사용한다.

- IMU: `/mavros/imu/data`
- GT pose: `/pose_transformed`

KAIST VIO에는 benchmark에서 바로 쓸 GNSS topic이 없기 때문에, 기본 설정에서는 GT pose를 일정 stride로 샘플링해서 `pseudo-position measurement`로 사용한다.

## Common Conversion

모든 benchmark는 먼저 `utils.prepare_dataset.prepare_dataset()`를 호출한다.

흐름은 대략 다음과 같다.

```text
KAIST rosbag
  -> IMU control: [ax, ay, az, gx, gy, gz]
  -> GT pose: [x, y, z, roll, pitch, yaw]
  -> optional pseudo-position: sampled GT [x, y, z]
  -> generated dataset CSV
```

생성되는 공통 dataset CSV는 각 output 폴더 아래에 저장된다.

예:

```text
outputs/benchmarks/<benchmark_name>/infinite/kaist_vio_infinite_dataset.csv
```

이 CSV는 Python filter들이 직접 쓰기 좋은 내부 benchmark format이다. C++ compare repo들은 이 format을 그대로 받지 않기 때문에, repo별 exporter/runner가 한 번 더 변환한다.

## Why InEKF Inputs Differ

EKF/UKF/PF 비교는 대부분 `position measurement`를 공통 measurement로 두고 비교할 수 있다.

하지만 InEKF compare repo 두 개는 원래 설계된 measurement interface가 다르다.

| Repo | Native correction input | KAIST pseudo-position 처리 |
| --- | --- | --- |
| `compare_repos/invariant-ekf` | GPS/position correction | pseudo-position을 synthetic LLA GPS로 변환 |
| `compare_repos/drift` | velocity correction / kinematics correction | pseudo-position을 미분해서 velocity measurement로 변환 |

그래서 두 repo 모두 InEKF 계열이지만, 같은 `[x, y, z]` measurement를 그대로 넣는 방식은 공정하지 않다.

`invariant-ekf`는 원래 `addGps()` correction path가 있으므로 position/GPS 계열 입력을 쓰는 것이 자연스럽다. 반면 `drift`는 원래 velocity correction을 중심으로 동작하므로, KAIST GT position에서 velocity를 추정해서 `VelocityCorrection`으로 넣는 것이 repo-native한 비교에 가깝다.

## Benchmark Scripts

### `filterpy_kaist_vio_benchmark.py`

비교 대상:

- Our EKF
- Our UKF
- FilterPy EKF
- FilterPy UKF

입력 흐름:

```text
KAIST rosbag
  -> prepare_dataset()
  -> dataset samples with IMU control + sampled pseudo-position
  -> Python EKF/UKF state-space benchmark
```

FilterPy는 C++ repo가 아니므로 별도 repo input folder를 만들지 않는다. 같은 Python dataset sample을 local comparable filter와 FilterPy wrapper가 공유한다.

### `stonesoup_kaist_vio_benchmark.py`

비교 대상:

- Our EKF / UKF / PF
- Stone Soup EKF / UKF / PF

입력 흐름:

```text
KAIST rosbag
  -> prepare_dataset()
  -> dataset samples with IMU control + sampled pseudo-position
  -> Stone Soup Detection / Predictor / Updater input
```

Stone Soup도 Python library이므로 C++ repo용 입력 변환은 없다. pseudo-position은 Stone Soup measurement object로 감싸서 update에 사용한다.

### `invariant_ekf_kaist_vio_benchmark.py`

비교 대상:

- Our InEKF 9D
- Our InEKF 15D
- `compare_repos/invariant-ekf`

입력 흐름:

```text
KAIST rosbag
  -> prepare_dataset()
  -> common dataset CSV
  -> our_inekf_9d.csv
  -> our_inekf_15d.csv
  -> repo_inputs/invariant_ekf/Log Files/*.csv
```

외부 repo 입력은 `utils.cpp_repo_exporters.export_invariant_ekf_input()`에서 만든다.

`invariant-ekf` repo가 원래 찾는 파일 구조는 다음과 같다.

```text
repo_inputs/invariant_ekf/
  Log Files/
    OnboardPose.csv
    OnboardGPS.csv
    GroundTruthAGL.csv
```

변환 방식:

- `OnboardPose.csv`: IMU timestamp, gyro, accel, accel bias를 기록한다.
- `OnboardGPS.csv`: KAIST local pseudo-position `[x, y, z]`를 fixed origin 주변 synthetic LLA GPS로 변환한다.
- `GroundTruthAGL.csv`: native folder shape 유지를 위해 생성한다.

즉 이 benchmark에서는 `invariant-ekf`의 native GPS/position correction path를 사용한다.

### `drift_kaist_vio_benchmark.py`

비교 대상:

- Our InEKF 9D with velocity update
- Our InEKF 15D with velocity update
- `compare_repos/drift`

입력 흐름:

```text
KAIST rosbag
  -> prepare_dataset()
  -> common dataset CSV
  -> GT position derivative -> velocity measurement
  -> our_inekf_9d_velocity.csv
  -> our_inekf_15d_velocity.csv
  -> repo_inputs/drift.csv
```

외부 repo 입력은 `utils.cpp_repo_exporters.export_drift_input()`에서 만든다.

`drift.csv`에는 다음 계열 값들이 들어간다.

- timestamp
- orientation quaternion
- angular velocity
- linear acceleration
- estimated velocity `[vel_x, vel_y, vel_z]`
- GT pose

`drift` runner는 이 CSV를 읽어서 다음 순서로 실행한다.

```text
ImuMeasurement
  -> ImuPropagation.Propagate()
VelocityMeasurement
  -> VelocityCorrection.Correct()
```

이제 `drift` benchmark는 position adapter를 쓰지 않는다. KAIST pseudo-position은 직접 position correction으로 들어가지 않고, GT position을 미분한 velocity measurement로 변환되어 `drift`의 native velocity correction에 들어간다.

## Output Files

각 benchmark는 보통 다음 파일들을 만든다.

```text
outputs/benchmarks/<benchmark_name>/infinite/
  *_results.csv
  metadata.json
  rmse_position.png
  error_variance.png
  runtime_sec.png
  trajectory_comparison_*.mp4
  estimates/*.csv
```

C++ compare repo benchmark는 추가로 다음 입력 snapshot을 남긴다.

```text
filter_inputs/
repo_inputs/
```

`filter_inputs/`는 우리 Python filter가 실제로 받은 입력을 확인하기 위한 CSV이고, `repo_inputs/`는 외부 C++ runner가 읽는 repo-specific 입력이다.
