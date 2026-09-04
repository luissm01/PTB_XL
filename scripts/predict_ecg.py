"""Run reproducible inference on one compatible WFDB ECG record."""

import argparse
from pathlib import Path

from ptbxl.experiments import get_clean_git_commit
from ptbxl.inference import load_inference_config, run_single_record_inference


DEFAULT_CONFIG_PATH = Path("configs/inference_small_cnn_100hz.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record_path", type=Path, help="WFDB basename, without extension"
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--record-id")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    git_commit = get_clean_git_commit(Path.cwd())
    config = load_inference_config(args.config_path)
    report = run_single_record_inference(
        config,
        args.config_path,
        args.record_path,
        args.output_path,
        git_commit,
        record_id=args.record_id,
    )
    positives = [item["label"] for item in report["predictions"] if item["predicted"]]
    print(f"Record: {report['input']['record_id']}")
    for item in report["predictions"]:
        print(
            f"{item['label']}: probability={item['probability']:.6f}, "
            f"threshold={item['threshold']:.6f}, predicted={item['predicted']}"
        )
    print(f"Positive labels: {', '.join(positives) if positives else 'none'}")
    print(f"Report: {args.output_path}")


if __name__ == "__main__":
    main()
