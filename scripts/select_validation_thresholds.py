"""Select and freeze baseline thresholds using validation only."""

import argparse
from pathlib import Path

from ptbxl.experiments import (
    get_clean_git_commit,
    load_threshold_experiment_config,
    run_validation_threshold_selection,
)


DEFAULT_CONFIG_PATH = Path("configs/baseline_small_cnn_100hz_thresholds.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = get_clean_git_commit(Path.cwd())
    config = load_threshold_experiment_config(args.config_path)
    print(
        f"Selecting {config.name} from commit {git_commit[:7]} on "
        f"{config.device}; test remains sealed.",
        flush=True,
    )
    artifact = run_validation_threshold_selection(
        config,
        args.config_path,
        git_commit,
    )
    thresholds = artifact["selection"]["thresholds"]
    print(
        "Thresholds: "
        + ", ".join(f"{label}={value:.6f}" for label, value in thresholds.items()),
        flush=True,
    )
    print(
        f"Validation macro F1: {artifact['validation']['macro']['f1']:.6f}; "
        f"micro F1: {artifact['validation']['micro']['f1']:.6f}; "
        f"artifact: {config.artifact_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
