"""Build and audit the initial PTB-XL five-superclass modeling cohort."""

import argparse
from pathlib import Path

import pandas as pd

from ptbxl.data.cohort import (
    build_cohort_summary,
    build_modeling_cohort,
    find_cohort_exclusions,
)
from ptbxl.data.reporting import compute_sha256, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", required=True, type=Path)
    parser.add_argument("--cohort-output-path", required=True, type=Path)
    parser.add_argument("--exclusions-output-path", required=True, type=Path)
    parser.add_argument("--report-output-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_path)
    cohort = build_modeling_cohort(labels)
    exclusions = find_cohort_exclusions(labels)
    report = {
        "source_labels_sha256": compute_sha256(args.labels_path),
        **build_cohort_summary(labels, cohort),
    }

    args.cohort_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.exclusions_output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.cohort_output_path, index=False, lineterminator="\n")
    exclusions.to_csv(args.exclusions_output_path, index=False, lineterminator="\n")
    write_json_report(report, args.report_output_path)
    print(
        f"Built cohort with {len(cohort)} included and {len(exclusions)} excluded; "
        f"report: {args.report_output_path}"
    )


if __name__ == "__main__":
    main()
