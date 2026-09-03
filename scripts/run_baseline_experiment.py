"""Run the configured PTB-XL train/validation baseline from a clean Git tree."""

import argparse
from pathlib import Path

from ptbxl.experiments import (
    get_clean_git_commit,
    load_experiment_config,
    run_baseline_experiment,
)
from ptbxl.training import FitEpochResult


DEFAULT_CONFIG_PATH = Path("configs/baseline_small_cnn_100hz.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def print_epoch(result: FitEpochResult) -> None:
    """Print compact progress without changing the persisted result."""
    print(
        f"epoch={result.epoch} "
        f"train_loss={result.train.loss:.6f} "
        f"validation_loss={result.validation.loss:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    git_commit = get_clean_git_commit(Path.cwd())
    config = load_experiment_config(args.config_path)
    print(
        f"Running {config.name} from commit {git_commit[:7]} on {config.device}; "
        f"test remains sealed.",
        flush=True,
    )
    report = run_baseline_experiment(
        config,
        args.config_path,
        git_commit,
        on_epoch_end=print_epoch,
    )
    print(
        f"Best epoch: {report['training']['best_epoch']}; "
        f"validation loss: {report['training']['best_validation_loss']:.6f}; "
        f"macro AUROC: {report['validation']['macro_auroc']:.6f}; "
        f"macro AUPRC: {report['validation']['macro_auprc']:.6f}; "
        f"report: {config.report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
