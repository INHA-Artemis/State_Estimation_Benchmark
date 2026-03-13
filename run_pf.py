# [협업 주석]
# Goal: PF benchmark 실행용 entry script를 제공한다.
# What it does: config(YAML) 로드, dataset/model/filter 생성, mode 분기(imu_only/gps_only/fused),
# time-step loop(predict/update), estimate 수집 및 후속 evaluation/visualization 연동 포인트를 제공한다.
"""Entry point for running Particle Filter benchmark pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core.runner import run_estimation_benchmark
from utils.config_loader import load_yaml_config

VALID_MODES = {"imu_only", "gps_only", "fused"}


def run_pf(
    pf_config_path: Path,
    dataset_config_path: Path,
    mode_override: str | None = None,
    filter_override: str | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Run one configured estimator (PF now, EKF/InEKF later via filter config)."""
    pf_cfg = load_yaml_config(pf_config_path)
    ds_cfg = load_yaml_config(dataset_config_path)

    if filter_override:
        pf_cfg.setdefault("filter", {})
        pf_cfg["filter"]["name"] = filter_override.lower()

    if mode_override and mode_override not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got '{mode_override}'")

    run = run_estimation_benchmark(
        pf_cfg=pf_cfg,
        dataset_cfg=ds_cfg,
        mode_override=mode_override,
        output_dir_override=output_dir_override,
    )

    metrics = run["metrics"]
    print(f"[{run['filter_name'].upper()}] mode={run['mode']}, pose={run['pose_type']}, state_dim={run['state_dim']}")
    if "position_rmse" in metrics:
        print(
            "[Metrics] "
            f"position_rmse={metrics['position_rmse']:.4f} m, "
            f"state_rmse={metrics['state_rmse']:.4f}, "
            f"latency_mean={metrics['latency']['step']['mean_ms']:.3f} ms"
        )
    else:
        print(f"[Metrics] latency_mean={metrics['latency']['step']['mean_ms']:.3f} ms (GT unavailable)")
    print(f"[Artifacts] output_dir={run['output_dir']}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Particle Filter benchmark pipeline")
    parser.add_argument("--pf-config", type=Path, default=Path("config/pf.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("config/dataset_config.yaml"))
    parser.add_argument("--mode", type=str, default=None, choices=sorted(VALID_MODES))
    parser.add_argument("--filter", type=str, default=None, help="Filter name override (pf/ekf/inekf).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    args = parser.parse_args()

    run_pf(
        pf_config_path=args.pf_config,
        dataset_config_path=args.dataset_config,
        mode_override=args.mode,
        filter_override=args.filter,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
