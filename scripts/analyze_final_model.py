"""Create the immutable post-hoc error analysis and portfolio figures."""

import argparse
from pathlib import Path

from ptbxl.experiments import (
    get_clean_git_commit,
    load_final_analysis_config,
    run_final_analysis,
)


DEFAULT_CONFIG_PATH = Path("configs/final_analysis_small_cnn_100hz.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = get_clean_git_commit(Path.cwd())
    config = load_final_analysis_config(args.config_path)
    print(
        f"Analyzing saved fold-10 predictions from commit {git_commit[:7]}; "
        "the frozen model and thresholds will not change.",
        flush=True,
    )
    report = run_final_analysis(config, args.config_path, git_commit)
    errors = report["error_analysis"]
    attribution = report["attribution"]
    print(
        f"Exact-match rate: {errors['exact_match_rate']:.6f}; "
        f"Hamming loss: {errors['hamming_loss']:.6f}"
    )
    print(
        f"Explained ECG {attribution['record']['ecg_id']} for "
        f"{attribution['label']}; top leads: "
        f"{', '.join(attribution['top_leads'])}"
    )
    print(f"Report: {config.report_path}")
    print(f"Figures: {config.figure_directory}")


if __name__ == "__main__":
    main()
