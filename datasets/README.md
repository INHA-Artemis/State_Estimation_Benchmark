# Dataset Guide

Dataset selection examples and dataset-specific notes live here instead of the top-level README.
The shared dataset config file is `config/dataset_config.yaml`.

## Common Fields

Key fields:
- `dataset_type`
- `dataset_name`
- `pose_type`
- `mode`
- `generated_csv_path`

Runtime modes:
- `imu_only`: prediction only
- `gnss_only`: measurement update only
- `fused`: prediction + measurement update

## Dataset Examples

Synthetic example:

```yaml
dataset_type: synthetic
dataset_name: synthetic
pose_type: 2d
mode: fused
sequence_length: 300
dt: 0.1
seed: 10
```

EuRoC example:

```yaml
dataset_type: euroc
euroc_imu_csv: /path/to/euroc/mav0/imu0/data.csv
euroc_gt_csv: /path/to/euroc/mav0/state_groundtruth_estimate0/data.csv
```

ROS bag example:

```yaml
dataset_type: rosbag
rosbag_path: /path/to/your_rosbag
rosbag_imu_topic: /mavros/imu/data
rosbag_gt_topic: /pose_transformed
```

M2DGR example:

```yaml
dataset_type: m2dgr
m2dgr_bag_path: /m2dgr_dataset/M2DGR/street_01.bag
m2dgr_gt_txt_path: /m2dgr_dataset/M2DGR/street_01.txt
m2dgr_imu_topic: /handsfree/imu
m2dgr_gnss_topic: /ublox/fix
```

## References

- EuRoC / Newer College style reference: https://ori.ox.ac.uk/datasets/newer-college-dataset
- KAIST-VIO: https://github.com/url-kaist/kaistviodataset?tab=readme-ov-file#Dataset-format
- InCrowd-VI: https://vault.cloudlab.zhaw.ch/vaults/InCrowd-VI/data/
- GVINS: https://github.com/HKUST-Aerial-Robotics/GVINS-Dataset
- M2DGR: https://github.com/SJTU-ViSYS/M2DGR?tab=readme-ov-file#dataset-sequences
