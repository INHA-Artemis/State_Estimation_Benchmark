# IMU/GNSS State Estimation Benchmark Library

This repository is a Python benchmark library for IMU/GNSS state estimation algorithms.
It provides a unified pipeline for loading datasets, generating a common CSV representation, running a filter, and saving plots, error curves, animations, and estimate CSV files.

## Overview

Implemented filters:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

Current state representations:
- `2d`: `[x, y, yaw]`
- `6d`: `[x, y, z, roll, pitch, yaw]`

Supported dataset sources:
- `synthetic`
- `euroc`
- `rosbag`
- `m2dgr`

## Implementation Notes

The EKF, UKF, and PF implementations in this repository are written with **NumPy-based array operations**.
This is the same design direction used by the particle filter path in this repository.

What that means in practice:
- state, covariance, particles, measurements, and controls are handled as NumPy arrays
- linear algebra is computed with `numpy.linalg`
- dataset preparation and evaluation also use NumPy arrays end-to-end
- no external Kalman filtering library is required for EKF or UKF execution

The EKF and UKF were intentionally kept simple to match the repository style used by PF:
- reuse existing files under `utils`, `models`, and `datasets`
- keep the runner/output flow identical to PF
- save the same kinds of result files with only the estimator name changed

## Repository Structure

Main files for Kalman filters:
- `config/ekf.yaml`
- `config/ukf.yaml`
- `examples/run_ekf.py`
- `examples/run_ukf.py`
- `filters/estimated_kalman_filter.py`
- `filters/unscented_kalman_filter.py`

Shared files reused by PF, EKF, and UKF:
- `config/dataset_config.yaml`
- `datasets/euroc_loader.py`
- `datasets/m2dgr_loader.py`
- `datasets/rosbag_loader.py`
- `utils/csv_dataset.py`
- `utils/generate_gnss.py`
- `utils/generate_imu.py`
- `utils/math_utils.py`
- `utils/save_estimates.py`
- `utils/visualization.py`
- `utils/yaml_loader.py`
- `models/state_model.py`
- `models/measurement_model.py`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Execution Flow

All filters follow the same high-level flow:
1. Read `config/dataset_config.yaml`
2. Read the filter-specific config file
3. Load or generate controls, measurements, ground truth, and timestamps
4. Save a unified dataset CSV
5. Run the filter
6. Save estimate CSV, trajectory plot, error plot, and optional animation

## How To Run Kalman Filters

Run EKF:

```bash
cd /workspace/State_Estimation_Benchmark
python3 examples/run_ekf.py
```

Run UKF:

```bash
cd /workspace/State_Estimation_Benchmark
python3 examples/run_ukf.py
```

Run PF:

```bash
cd /workspace/State_Estimation_Benchmark
python3 examples/run_pf.py
```

Override config paths explicitly:

```bash
python3 examples/run_ekf.py \
  --dataset-config config/dataset_config.yaml \
  --ekf-config config/ekf.yaml \
  --output-dir outputs

python3 examples/run_ukf.py \
  --dataset-config config/dataset_config.yaml \
  --ukf-config config/ukf.yaml \
  --output-dir outputs

python3 examples/run_pf.py \
  --dataset-config config/dataset_config.yaml \
  --pf-config config/pf.yaml \
  --output-dir outputs
```

## Configuration

### Dataset Config

The shared dataset configuration is:
- `config/dataset_config.yaml`

Important fields:
- `dataset_type`: `synthetic`, `euroc`, `rosbag`, `m2dgr`
- `dataset_name`: used in output filenames
- `pose_type`: `2d` or `6d`
- `mode`: `imu_only`, `gnss_only`, `fused`
- `generated_csv_path`: path to the unified dataset CSV

Dataset-specific sections are already included in the file for:
- EuRoC
- ROS bag
- M2DGR
- synthetic generation

### Filter Configs

PF:
- `config/pf.yaml`

EKF:
- `config/ekf.yaml`

UKF:
- `config/ukf.yaml`

Common filter config sections:
- `initialization`
- `motion_model`
- `measurement_model`
- `evaluation`
- `visualization`
- `output`

UKF-only section:
- `sigma_points`

## Dataset Examples

### EuRoC

Example dataset config values:

```yaml
dataset_type: euroc
dataset_name: euroc
pose_type: 6d
mode: fused
generated_csv_path: outputs/euroc.csv

euroc_imu_csv: /path/to/euroc/mav0/imu0/data.csv
euroc_gt_csv: /path/to/euroc/mav0/state_groundtruth_estimate0/data.csv
euroc_use_gt_as_gnss: true

gnss_noise_std: [0.7, 0.7, 0.7]
```

Run:

```bash
python3 examples/run_ekf.py
python3 examples/run_ukf.py
python3 examples/run_pf.py
```

### M2DGR

Example dataset config values:

```yaml
dataset_type: m2dgr
dataset_name: street_01
pose_type: 6d
mode: fused
generated_csv_path: outputs/m2dgr.csv

m2dgr_bag_path: /m2dgr_dataset/M2DGR/street_01.bag
m2dgr_gt_txt_path: /m2dgr_dataset/M2DGR/street_01.txt
m2dgr_imu_topic: /handsfree/imu
m2dgr_gnss_topic: /ublox/fix
m2dgr_linear_source: gt_velocity
m2dgr_use_gt_as_gnss: false

gnss_noise_std: [0.7, 0.7, 0.7]
```

Run:

```bash
python3 examples/run_ekf.py
python3 examples/run_ukf.py
python3 examples/run_pf.py
```

### ROS Bag

Example dataset config values:

```yaml
dataset_type: rosbag
dataset_name: kaistvio
pose_type: 6d
mode: fused

rosbag_path: /path/to/your_rosbag
rosbag_imu_topic: /mavros/imu/data
rosbag_gt_topic: /pose_transformed
rosbag_linear_source: gt_velocity
rosbag_use_gt_as_gnss: true
```

Supported path formats:
- ROS1 `.bag`
- ROS2 bag directory
- ROS2 `metadata.yaml`
- ROS2 `.db3`

### Synthetic

Example dataset config values:

```yaml
dataset_type: synthetic
dataset_name: synthetic
pose_type: 2d
mode: fused
sequence_length: 300
dt: 0.1
seed: 10
```

## Output Files

For all three filters, the output style is the same.
Only the estimator prefix changes.

Common outputs:
- unified dataset CSV
- estimates CSV
- trajectory PNG
- position error norm PNG
- optional mp4 or gif animation

Examples:
- `outputs/<dataset_name>_ekf_estimates.csv`
- `outputs/<dataset_name>_ekf_trajectory.png`
- `outputs/<dataset_name>_ekf_position_error_norm.png`
- `outputs/<dataset_name>_ekf_trajectory.mp4`
- `outputs/<dataset_name>_ukf_estimates.csv`
- `outputs/<dataset_name>_ukf_trajectory.png`
- `outputs/<dataset_name>_ukf_position_error_norm.png`
- `outputs/<dataset_name>_ukf_trajectory.mp4`
- `outputs/<dataset_name>_pf_estimates.csv`
- `outputs/<dataset_name>_pf_trajectory.png`
- `outputs/<dataset_name>_pf_position_error_norm.png`
- `outputs/<dataset_name>_pf_trajectory.mp4`

## Console Output

When a runner finishes, it prints:
- pose type
- dataset CSV path
- number of steps
- position RMSE
- filter runtime
- output file paths

## Notes On EKF And UKF Behavior

The EKF and UKF implementations in this repository are benchmark-oriented, not full external-framework reproductions.
They are designed to fit the repository's existing pipeline and keep the same interface as PF.

In particular:
- they reuse the repository's current `2d` and `6d` state conventions
- they use the same dataset loaders and CSV conversion flow as PF
- they normalize measurement config automatically for 3D runs when needed
- they keep outputs identical in structure to PF outputs

## References

The implementation is inspired by existing libraries and tutorials:
- [navlie](https://github.com/decargroup/navlie)
- [FilterPy](https://github.com/rlabbe/filterpy)
- [Stone Soup tutorials](https://stonesoup.readthedocs.io/en/v1.2/auto_tutorials/index.html)
- [robot_localization](https://github.com/cra-ros-pkg/robot_localization)
- [DRIFT](https://github.com/UMich-CURLY/drift)
- Roger Labbe, *Kalman and Bayesian Filters in Python*, Chapter 12 Particle Filters
