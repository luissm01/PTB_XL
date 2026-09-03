"""Execute the sealed PTB-XL fold-10 evaluation exactly once."""

import argparse
from pathlib import Path

from ptbxl.experiments import (
    get_clean_git_commit,
    load_final_test_config,
    run_final_test_evaluation,
)


DEFAULT_CONFIG_PATH = Path("configs/final_test_small_cnn_100hz.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = get_clean_git_commit(Path.cwd())
    config = load_final_test_config(args.config_path)
    print(
        f"Executing immutable fold-10 event {config.name} from commit "
        f"{git_commit[:7]} on {config.device}. This command cannot be repeated.",
        flush=True,
    )
    report = run_final_test_evaluation(config, args.config_path, git_commit)
    print(
        f"Final test samples: {report['dataset']['samples']}; "
        f"macro AUROC: {report['ranking']['macro_auroc']:.6f}; "
        f"macro AUPRC: {report['ranking']['macro_auprc']:.6f}; "
        f"macro F1: {report['operating_point']['macro']['f1']:.6f}; "
        f"report: {config.report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
