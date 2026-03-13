# [협업 주석]
# Goal: PF 실행 예제를 독립적으로 제공한다.
# What it does: sample config 또는 CLI 기반으로 PF 파이프라인 실행 예시를 담을 예정이다.
"""Example script for running PF benchmark from the examples directory."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """
    Goal:
        examples 디렉터리에서 PF benchmark를 바로 실행하는 진입점을 제공한다.
    Input:
        별도 argument 없이 repo_root 기준 기본 config 경로를 내부에서 구성한다.
    Output:
        없음. run_pf를 호출해 benchmark를 실행한다.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from run_pf import run_pf

    run_pf(
        pf_config_path=repo_root / "config" / "pf.yaml",
        dataset_config_path=repo_root / "config" / "dataset_config.yaml",
        mode_override=None,
    )


if __name__ == "__main__":
    main()
