from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# examples/에서 실행해도 루트 모듈 import가 되도록 프로젝트 루트를 sys.path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.euroc_loader import load_euroc_dataset
from filters.particle_filter import ParticleFilter
from utils.csv_dataset import load_dataset_from_csv, save_dataset_to_csv
from utils.generate_gnss import generate_gnss_measurements
from utils.generate_imu import generate_imu_controls
from utils.math_utils import compute_rmse
from utils.visualization import plot_results, save_trajectory_animation
from utils.yaml_loader import load_yaml


def main() -> None:
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run Particle Filter and plot trajectory.")
    parser.add_argument("--dataset-config", default=str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--pf-config", default=str(PROJECT_ROOT / "config" / "pf.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    # 1) 실행에 필요한 dataset / filter 설정 파일을 읽는다.
    dataset_cfg = load_yaml(Path(args.dataset_config))
    pf_cfg = load_yaml(Path(args.pf_config))

    # 2) 상태 표현 차원을 정규화한다.
    #    config에서 6d라고 쓰면 내부 PF는 3d(x, y, z, roll, pitch, yaw) 표현으로 처리한다.
    pose_type = dataset_cfg.get("pose_type", "2d")
    if pose_type == "6d":
        pose_type = "3d"

    # 3) 3D/6D 실행이면 measurement model이 최소 x,y,z를 보도록 보정한다.
    #    EuRoC나 pseudo-GNSS 측정은 3축 위치를 사용하므로, position_indices와 noise 길이를 맞춘다.
    if pose_type == "3d":
        measurement_cfg = pf_cfg.setdefault("measurement_model", {})
        position_indices = list(measurement_cfg.get("position_indices", [0, 1]))
        if len(position_indices) < 3:
            measurement_cfg["position_indices"] = [0, 1, 2]

        measurement_noise = list(measurement_cfg.get("measurement_noise_diag", [0.7, 0.7]))
        if len(measurement_noise) == 1:
            measurement_noise = measurement_noise * 3
        elif len(measurement_noise) == 2:
            measurement_noise.append(measurement_noise[-1])
        measurement_cfg["measurement_noise_diag"] = measurement_noise

    # 4) 실행 모드를 정규화한다.
    #    코드 내부에서는 gps_only 대신 gnss_only라는 이름을 사용한다.
    mode = dataset_cfg.get("mode", "fused")
    if mode == "gps_only":
        mode = "gnss_only"
        dataset_cfg["mode"] = mode

    # 5) 중간 benchmark CSV 저장 경로를 절대경로로 만든다.
    #    이후 모든 PF 실행은 이 통합 CSV 포맷을 기준으로 진행된다.
    generated_csv_path = Path(dataset_cfg.get("generated_csv_path", PROJECT_ROOT / "outputs" / f"synthetic_{pose_type}.csv"))
    if not generated_csv_path.is_absolute():
        generated_csv_path = PROJECT_ROOT / generated_csv_path
    dataset_cfg["generated_csv_path"] = generated_csv_path

    # 6) 원본 데이터를 준비한다.
    #    - synthetic: 코드에서 IMU / GNSS / GT를 생성한다.
    #    - euroc: EuRoC IMU + GT를 읽어 PF가 이해하는 control / measurement / gt / dt로 변환한다.
    dataset_type = dataset_cfg.get("dataset_type", "synthetic")
    if dataset_type == "euroc":
        pose_type = "3d"
        dataset_cfg["pose_type"] = "6d"
        controls, measurements, gt, dt = load_euroc_dataset(dataset_cfg)
    else:
        controls, gt = generate_imu_controls(dataset_cfg, pose_type=pose_type)
        measurements = generate_gnss_measurements(dataset_cfg, pose_type=pose_type, gt=gt)
        dt = float(dataset_cfg.get("dt", 0.1))

    # 7) 모든 데이터를 통합 CSV로 저장한다.
    #    이 CSV는 한 step당 하나의 control / measurement / gt / dt를 담는다.
    csv_path = save_dataset_to_csv(
        generated_csv_path,
        pose_type=pose_type,
        dt=dt,
        controls=controls,
        measurements=measurements,
        gt=gt,
    )

    # 8) 통합 CSV를 다시 읽어 PF 입력 포맷(list[dict])으로 변환한다.
    #    각 row는 {control, measurement, dt, gt} 구조를 가진다.
    dataset, gt = load_dataset_from_csv(csv_path, pose_type=pose_type, mode=mode)

    # 9) PF 객체를 config로부터 생성하고 초기 particle들을 샘플링한다.
    #    initialization.mean / cov_diag를 기준으로 particle set이 초기화된다.
    pf = ParticleFilter.from_configs(dataset_cfg, pf_cfg)

    # 10) PF를 전체 시퀀스에 대해 실행한다.
    #     내부적으로 각 step마다 아래 순서를 반복한다.
    #     a. predict(control, dt): 현재 control로 모든 particle을 전파한다.
    #     b. measurement_update(measurement): 관측과의 오차로 particle weight를 갱신한다.
    #     c. effective_sample_size()가 threshold보다 작으면 resampling한다.
    #        즉, weight가 일부 particle에만 몰렸을 때 particle set을 다시 균등하게 재구성한다.
    #     d. estimate_pose(): resampled/updated particle들의 가중평균으로 현재 pose를 출력한다.
    pf_start = time.perf_counter()
    estimates = pf.run(dataset)
    pf_runtime = time.perf_counter() - pf_start

    # 11) 추정 결과와 GT 사이의 위치 RMSE를 계산해 콘솔에 출력한다.
    rmse = compute_rmse(estimates, gt, pose_type=pose_type)
    print(f"[PF] Pose type: {pose_type}")
    print(f"[PF] Dataset CSV: {csv_path}")
    print(f"[PF] Steps: {len(dataset)}")
    print(f"[PF] RMSE (position): {rmse:.4f}")
    print(f"[PF] Runtime (filter only): {pf_runtime:.3f} sec")

    # 12) 정적 PNG 결과를 저장한다.
    #     시각화 옵션은 pf.yaml의 visualization 섹션에서 관리한다.
    out_dir = Path(args.output_dir)
    plot_path = out_dir / f"pf_trajectory_{pose_type}.png"
    plot_results(estimates, gt, pose_type=pose_type, save_path=plot_path, visual_cfg=pf_cfg.get("visualization", {}))
    print(f"[PF] Plot saved: {plot_path}")

    # 13) 설정이 허용되면 MP4 애니메이션도 생성한다.
    #     애니메이션은 전체 궤적 대신 최근 tail_length step만 보여줘 정적 PNG보다 해석이 쉽다.
    vis_cfg = pf_cfg.get("visualization", {})
    anim_cfg = vis_cfg.get("animation", {})
    if vis_cfg.get("save_animation", False) and anim_cfg.get("format", "mp4") == "mp4":
        video_path = out_dir / f"pf_trajectory_{pose_type}.mp4"
        saved = save_trajectory_animation(
            estimates,
            gt,
            pose_type=pose_type,
            save_path=video_path,
            fps=int(anim_cfg.get("fps", 20)),
            tail_length=int(anim_cfg.get("tail_length", 80)),
        )
        if saved:
            print(f"[PF] Animation saved: {video_path}")
        else:
            print("[PF] Animation skipped: ffmpeg writer is not available.")

    total_runtime = time.perf_counter() - total_start
    print(f"[PF] Runtime (total): {total_runtime:.3f} sec")


if __name__ == "__main__":
    main()
