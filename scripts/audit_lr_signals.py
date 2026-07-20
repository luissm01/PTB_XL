"""Audit PTB-XL v1.0.3 low-resolution signals for the modeling cohort."""

import argparse
from pathlib import Path

import pandas as pd

from ptbxl.data.reporting import compute_sha256, write_json_report
from ptbxl.data.signals import audit_lr_signals


EXPECTED_COHORT_RECORDS = 21_388


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-path", required=True, type=Path)
    parser.add_argument(
        "--metadata-path",
        default=Path("data/raw/ptbxl_database.csv"),
        type=Path,
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--signal-manifest-path",
        default=Path("data/raw/SHA256SUMS.txt"),
        type=Path,
    )
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument(
        "--expected-records",
        default=EXPECTED_COHORT_RECORDS,
        type=int,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(args.cohort_path)
    if len(cohort) != args.expected_records:
        raise ValueError(
            f"Expected {args.expected_records} cohort records, got {len(cohort)}"
        )
    metadata = pd.read_csv(
        args.metadata_path,
        usecols=["ecg_id", "filename_lr"],
    )
    report = {
        "sources": {
            "cohort_sha256": compute_sha256(args.cohort_path),
            "metadata_sha256": compute_sha256(args.metadata_path),
            "signal_manifest_sha256": compute_sha256(args.signal_manifest_path),
        },
        **audit_lr_signals(cohort, metadata, args.dataset_root),
    }
    write_json_report(report, args.output_path)
    records = report["records"]
    print(
        f"Audited {records['expected']} signals: {records['loaded']} loaded, "
        f"{records['missing']} missing, {records['invalid']} invalid; "
        f"report: {args.output_path}"
    )
    if records["missing"] or records["invalid"]:
        raise SystemExit("Signal audit found missing or invalid records")


if __name__ == "__main__":
    main()
